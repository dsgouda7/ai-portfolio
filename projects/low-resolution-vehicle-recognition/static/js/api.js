const JSON_HEADERS = { "Content-Type": "application/json", Accept: "application/json" };

export class ApiError extends Error {
  constructor(message, status = 0, detail = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function requestJson(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), options.timeout ?? 10000);
  try {
    const response = await fetch(path, {
      ...options,
      headers: { ...JSON_HEADERS, ...(options.headers || {}) },
      signal: options.signal || controller.signal,
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { message: text };
      }
    }
    if (!response.ok) {
      const message = payload?.error?.message || payload?.message || payload?.error || `${response.status} ${response.statusText}`;
      throw new ApiError(String(message), response.status, payload);
    }
    return payload ?? {};
  } catch (error) {
    if (error.name === "AbortError") {
      throw new ApiError("The CarFace server did not respond in time.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export const api = {
  status: () => requestJson("/api/status"),
  sources: () => requestJson("/api/sources"),
  startRun: (sourceId) => requestJson("/api/runs", {
    method: "POST",
    body: JSON.stringify({ source_id: sourceId }),
  }),
  run: (runId) => requestJson(`/api/runs/${encodeURIComponent(runId)}`),
  tracks: (runId) => requestJson(`/api/runs/${encodeURIComponent(runId)}/tracks`),
  track: (trackId) => requestJson(`/api/tracks/${encodeURIComponent(trackId)}`),
  transition: (runId, action) => requestJson(`/api/runs/${encodeURIComponent(runId)}/${action}`, {
    method: "POST",
    body: "{}",
  }),
};

export async function fetchCurrentFrame(runId) {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/frame?_=${Date.now()}`, {
    headers: { Accept: "image/jpeg, image/png, application/json" },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new ApiError(`Current frame unavailable (${response.status}).`, response.status);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.startsWith("image/")) {
    const blob = await response.blob();
    return {
      url: URL.createObjectURL(blob),
      objectUrl: true,
      frameId: response.headers.get("x-frame-id"),
      capturedAt: response.headers.get("x-captured-at"),
      overlays: [],
    };
  }

  const payload = await response.json();
  const frame = payload.frame || payload.current_frame || payload;
  let url = frame.url || frame.frame_url || frame.image_url || null;
  if (!url && frame.image_base64) {
    url = `data:${frame.content_type || "image/jpeg"};base64,${frame.image_base64}`;
  }
  return {
    url,
    objectUrl: false,
    frameId: frame.frame_id ?? payload.frame_id ?? null,
    capturedAt: frame.captured_at ?? payload.captured_at ?? null,
    width: frame.width ?? payload.width ?? null,
    height: frame.height ?? payload.height ?? null,
    overlays: frame.overlays || frame.detections || payload.overlays || [],
  };
}

function parseEventBlock(block) {
  const event = { type: "message", data: "", id: null, retry: null, heartbeat: false };
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) {
      event.heartbeat = true;
      continue;
    }
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event.type = value;
    if (field === "data") dataLines.push(value);
    if (field === "id" && !value.includes("\0")) event.id = value;
    if (field === "retry" && /^\d+$/.test(value)) event.retry = Number(value);
  }
  event.data = dataLines.join("\n");
  return event;
}

export class RunEventStream {
  constructor(runId, handlers = {}) {
    this.runId = runId;
    this.handlers = handlers;
    this.lastEventId = "";
    this.reconnectDelay = 1000;
    this.stopped = true;
    this.controller = null;
  }

  start() {
    if (!this.stopped) return;
    this.stopped = false;
    void this.#connectLoop();
  }

  stop() {
    this.stopped = true;
    this.controller?.abort();
    this.controller = null;
    this.handlers.onStatus?.("offline");
  }

  async #connectLoop() {
    while (!this.stopped) {
      this.controller = new AbortController();
      try {
        const headers = { Accept: "text/event-stream", "Cache-Control": "no-cache" };
        if (this.lastEventId) headers["Last-Event-ID"] = this.lastEventId;
        const response = await fetch(`/api/runs/${encodeURIComponent(this.runId)}/events`, {
          headers,
          cache: "no-store",
          signal: this.controller.signal,
        });
        if (!response.ok || !response.body) {
          throw new ApiError(`Event stream unavailable (${response.status}).`, response.status);
        }
        this.handlers.onStatus?.("live");
        await this.#readStream(response.body);
        if (this.stopped) return;
        const terminalRun = await this.#terminalRunAfterClose();
        if (terminalRun) {
          this.stopped = true;
          this.handlers.onTerminal?.(terminalRun);
          this.handlers.onStatus?.("complete");
          return;
        }
        throw new ApiError("Event stream closed before the run reached a terminal state.");
      } catch (error) {
        if (this.stopped) return;
        this.handlers.onStatus?.("reconnecting", error);
        await new Promise((resolve) => window.setTimeout(resolve, this.reconnectDelay));
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, 8000);
      }
    }
  }

  async #terminalRunAfterClose() {
    try {
      const payload = await api.run(this.runId);
      const run = payload.run || payload;
      return ["completed", "failed", "stopped"].includes(String(run.state).toLowerCase())
        ? run
        : null;
    } catch {
      return null;
    }
  }

  async #readStream(body) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (!this.stopped) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replaceAll("\r\n", "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        this.#dispatch(block);
        boundary = buffer.indexOf("\n\n");
      }
    }
  }

  #dispatch(block) {
    if (!block) return;
    const event = parseEventBlock(block);
    if (event.retry) this.reconnectDelay = Math.max(250, event.retry);
    if (event.id !== null) {
      this.lastEventId = event.id;
      this.reconnectDelay = 1000;
    }
    if (event.heartbeat || event.type === "heartbeat") {
      this.handlers.onHeartbeat?.(Date.now());
      return;
    }
    if (!event.data) return;
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      payload = { message: event.data };
    }
    this.handlers.onEvent?.(payload, event);
  }
}

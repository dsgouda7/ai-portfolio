import { ApiError, RunEventStream, api, fetchCurrentFrame } from "./api.js";
import { PipelineGraph, formatDuration } from "./pipeline_graph.js";

const elements = Object.fromEntries([
  "readiness-badge", "device-badge", "model-badge", "profile-badge", "source-select",
  "start-button", "pause-button", "resume-button", "stop-button", "run-state", "run-id",
  "connection-notice", "frame-image", "frame-empty", "frame-overlay", "live-indicator",
  "frame-number", "frame-age", "source-attribution", "tracks-body", "track-count",
  "pipeline-graph", "event-sequence", "trace-connection", "stage-detail-title",
  "stage-detail-status", "stage-detail-duration", "stage-detail-context",
  "stage-detail-message", "stage-detail-data", "selected-track-id", "evidence-strip",
  "hierarchy-panel", "abstention-panel", "toast-region",
].map((id) => [id, document.getElementById(id)]));

const state = {
  status: null,
  sources: [],
  runId: null,
  runState: "pending",
  selectedTrackId: null,
  eventSequence: null,
  eventStream: null,
  trackTimer: null,
  frameTimer: null,
  frameBusy: false,
  trackBusy: false,
  currentObjectUrl: null,
  framePollingSupported: true,
  lastFrameAt: null,
  pollGeneration: 0,
};

const graph = new PipelineGraph(elements["pipeline-graph"], renderStageDetail);

function listFrom(payload, key) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.[key])) return payload[key];
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

function text(value, fallback = "--") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function humanize(value) {
  return text(value, "Unknown").toLowerCase().replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatConfidence(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return `${Math.round(Math.max(0, Math.min(1, number)) * 100)}%`;
}

function setRunState(nextState) {
  const normalized = String(nextState || "pending").toLowerCase();
  state.runState = normalized;
  elements["run-state"].dataset.state = normalized;
  elements["run-state"].textContent = humanize(normalized);
  const active = ["pending", "running", "paused"].includes(normalized) && Boolean(state.runId);
  const running = normalized === "running";
  const paused = normalized === "paused";
  elements["start-button"].disabled = active || !elements["source-select"].value;
  elements["source-select"].disabled = active || state.sources.length === 0;
  elements["pause-button"].disabled = !running;
  elements["pause-button"].hidden = paused;
  elements["resume-button"].disabled = !paused;
  elements["resume-button"].hidden = !paused;
  elements["stop-button"].disabled = !active;
  elements["live-indicator"].hidden = !running;
}

function setTraceConnection(connection) {
  elements["trace-connection"].dataset.state = connection;
  elements["trace-connection"].textContent = humanize(connection);
}

function showNotice(message) {
  elements["connection-notice"].textContent = message;
  elements["connection-notice"].hidden = !message;
}

function showToast(message, tone = "info") {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.dataset.tone = tone;
  toast.textContent = message;
  elements["toast-region"].append(toast);
  window.setTimeout(() => toast.remove(), 6000);
}

function describeError(error) {
  if (error instanceof ApiError) return error.message;
  return error?.message || "Unexpected browser error.";
}

async function loadStatus() {
  try {
    const payload = await api.status();
    const status = payload.status || payload;
    state.status = status;
    const ready = Boolean(status.ready ?? status.worker_ready ?? status.model_ready);
    elements["readiness-badge"].dataset.tone = ready ? "ready" : (status.error ? "failed" : "warning");
    elements["readiness-badge"].textContent = ready ? "Ready" : (status.error ? "Unavailable" : "Not ready");
    elements["device-badge"].textContent = `Device ${text(status.device || status.compute_device, "unknown")}`;
    elements["model-badge"].textContent = `Model ${text(status.model_version || status.model?.version, "not loaded")}`;

    const rawProfile = String(status.profile_type || status.profile || status.mode || "").toLowerCase();
    const demo = status.demo_mode === true || ["demo", "mock", "fixture"].includes(rawProfile);
    const real = status.demo_mode === false || ["real", "real-model", "production"].includes(rawProfile);
    elements["profile-badge"].dataset.profile = demo ? "demo" : (real ? "real" : "unknown");
    elements["profile-badge"].textContent = demo
      ? "Demo profile · non-production"
      : real ? "Real-model profile" : "Profile not reported";
  } catch (error) {
    elements["readiness-badge"].dataset.tone = "failed";
    elements["readiness-badge"].textContent = "Server unavailable";
    elements["profile-badge"].textContent = "Profile unavailable";
    showToast(describeError(error), "error");
  }
}

async function loadSources() {
  try {
    const payload = await api.sources();
    state.sources = listFrom(payload, "sources").filter((source) => source.enabled !== false);
    elements["source-select"].replaceChildren();
    if (state.sources.length === 0) {
      elements["source-select"].append(new Option("No enabled sources", ""));
      elements["source-select"].disabled = true;
      return;
    }
    elements["source-select"].append(new Option("Select an approved source", ""));
    for (const source of state.sources) {
      const sourceId = source.source_id || source.id;
      const option = new Option(source.name || sourceId, sourceId);
      option.dataset.type = source.adapter_type || source.type || "unknown";
      elements["source-select"].append(option);
    }
    elements["source-select"].disabled = false;
  } catch (error) {
    elements["source-select"].replaceChildren(new Option("Sources unavailable", ""));
    elements["source-select"].disabled = true;
    showToast(describeError(error), "error");
  }
}

function selectedSource() {
  return state.sources.find((source) => (source.source_id || source.id) === elements["source-select"].value);
}

function updateSourceContext() {
  const source = selectedSource();
  elements["start-button"].disabled = !source || Boolean(state.runId && ["pending", "running", "paused"].includes(state.runState));
  elements["source-attribution"].textContent = source?.attribution || source?.name || "Source attribution appears here";
  if (elements["profile-badge"].dataset.profile === "unknown" && source) {
    const demoSource = ["replay", "demo"].includes(String(source.adapter_type || source.type).toLowerCase());
    elements["profile-badge"].dataset.profile = demoSource ? "demo" : "unknown";
    elements["profile-badge"].textContent = demoSource ? "Demo source · model profile unreported" : "Model profile not reported";
  }
}

function resetRunSurface() {
  graph.reset();
  state.eventSequence = null;
  state.selectedTrackId = null;
  state.framePollingSupported = true;
  elements["event-sequence"].textContent = "Sequence --";
  elements["tracks-body"].replaceChildren(emptyTableRow("Tracks will appear as vehicles are associated."));
  elements["track-count"].textContent = "0";
  resetEvidence();
  showNotice("");
}

async function startRun() {
  const sourceId = elements["source-select"].value;
  if (!sourceId) return;
  elements["start-button"].disabled = true;
  resetRunSurface();
  try {
    const payload = await api.startRun(sourceId);
    const run = payload.run || payload;
    state.runId = run.run_id || run.id;
    if (!state.runId) throw new ApiError("Run response did not include a run ID.");
    state.pollGeneration += 1;
    elements["run-id"].textContent = state.runId;
    setRunState(run.state || "running");
    connectEvents();
    startPolling();
  } catch (error) {
    state.runId = null;
    elements["run-id"].textContent = "No active run";
    setRunState("failed");
    showToast(describeError(error), "error");
  }
}

async function transitionRun(action) {
  if (!state.runId) return;
  const button = elements[`${action}-button`];
  if (button) button.disabled = true;
  const stopping = action === "stop";
  if (stopping) {
    state.pollGeneration += 1;
    stopLiveUpdates();
  }
  try {
    const payload = await api.transition(state.runId, action);
    const run = payload.run || payload;
    setRunState(run.state || (action === "stop" ? "stopped" : action === "pause" ? "paused" : "running"));
    if (stopping) setTraceConnection("offline");
  } catch (error) {
    showToast(describeError(error), "error");
    setRunState(state.runState);
    if (stopping && ["pending", "running", "paused"].includes(state.runState)) {
      state.pollGeneration += 1;
      connectEvents();
      startPolling();
    }
  }
}

// SSE ordering and frame polling have separate lifecycles so trace reconnects never stall imagery.
function connectEvents() {
  state.eventStream?.stop();
  state.eventStream = new RunEventStream(state.runId, {
    onStatus: (connection, error) => {
      setTraceConnection(connection);
      const sequenceGapVisible = elements["connection-notice"].textContent.startsWith("Event sequence gap");
      if (connection === "reconnecting" && !sequenceGapVisible) {
        showNotice(`Execution trace reconnecting: ${describeError(error)}`);
      }
      if (["live", "complete"].includes(connection) && !sequenceGapVisible) showNotice("");
    },
    onTerminal: async (run) => {
      setRunState(run.state);
      stopPollingTimers();
      state.eventStream = null;
      elements["live-indicator"].hidden = true;
      await refreshRunAndTracks();
    },
    onHeartbeat: () => {
      setTraceConnection("live");
    },
    onEvent: handlePipelineEvent,
  });
  state.eventStream.start();
}

function handlePipelineEvent(event) {
  const sequence = Number(event.sequence_id);
  if (Number.isFinite(sequence)) {
    if (state.eventSequence !== null && sequence <= state.eventSequence) return;
    if (state.eventSequence !== null && sequence > state.eventSequence + 1) {
      showNotice(`Event sequence gap detected: expected ${state.eventSequence + 1}, received ${sequence}. The trace may be incomplete.`);
    }
    state.eventSequence = sequence;
    elements["event-sequence"].textContent = `Sequence ${sequence}`;
  }
  graph.update(event);
  const runState = event.run_state || event.output_summary?.run_state;
  if (runState) setRunState(runState);
}

function startPolling() {
  stopPollingTimers();
  void pollRunAndTracks();
  void pollFrame();
  state.trackTimer = window.setInterval(pollRunAndTracks, 1200);
  state.frameTimer = window.setInterval(pollFrame, 750);
}

function stopPollingTimers() {
  window.clearInterval(state.trackTimer);
  window.clearInterval(state.frameTimer);
  state.trackTimer = null;
  state.frameTimer = null;
}

function stopLiveUpdates() {
  stopPollingTimers();
  state.eventStream?.stop();
  state.eventStream = null;
  elements["live-indicator"].hidden = true;
}

async function pollRunAndTracks() {
  if (!state.runId || state.trackBusy || ["stopped", "completed", "failed"].includes(state.runState)) return;
  await refreshRunAndTracks();
}

async function refreshRunAndTracks() {
  if (!state.runId || state.trackBusy) return;
  const generation = state.pollGeneration;
  state.trackBusy = true;
  try {
    const [runPayload, tracksPayload] = await Promise.all([api.run(state.runId), api.tracks(state.runId)]);
    if (generation !== state.pollGeneration) return;
    const run = runPayload.run || runPayload;
    if (run.state) setRunState(run.state);
    renderTracks(listFrom(tracksPayload, "tracks"));
    if (["stopped", "completed", "failed"].includes(String(run.state).toLowerCase())) stopLiveUpdates();
  } catch (error) {
    if (error.status !== 404) showToast(describeError(error), "error");
  } finally {
    state.trackBusy = false;
  }
}

async function pollFrame() {
  if (!state.runId || state.frameBusy || !state.framePollingSupported || state.runState !== "running") return;
  const generation = state.pollGeneration;
  state.frameBusy = true;
  try {
    const frame = await fetchCurrentFrame(state.runId);
    if (generation !== state.pollGeneration) {
      if (frame.objectUrl) URL.revokeObjectURL(frame.url);
      return;
    }
    if (!frame.url) return;
    if (state.currentObjectUrl) URL.revokeObjectURL(state.currentObjectUrl);
    state.currentObjectUrl = frame.objectUrl ? frame.url : null;
    elements["frame-image"].src = frame.url;
    elements["frame-image"].hidden = false;
    elements["frame-empty"].hidden = true;
    elements["frame-number"].textContent = `Frame ${text(frame.frameId)}`;
    state.lastFrameAt = frame.capturedAt ? new Date(frame.capturedAt) : new Date();
    elements["frame-age"].textContent = "Updated now";
    renderOverlays(frame);
  } catch (error) {
    if (error.status === 404 || error.status === 405) {
      state.framePollingSupported = false;
      elements["frame-age"].textContent = "Frame route unavailable";
    } else {
      elements["frame-age"].textContent = "Frame update delayed";
    }
  } finally {
    state.frameBusy = false;
  }
}

function renderOverlays(frame) {
  elements["frame-overlay"].replaceChildren();
  const width = Number(frame.width);
  const height = Number(frame.height);
  if (!(width > 0 && height > 0)) return;
  for (const overlay of frame.overlays || []) {
    const box = overlay.bbox_xyxy || overlay.bbox;
    if (!Array.isArray(box) || box.length !== 4) continue;
    const node = document.createElement("div");
    node.className = "overlay-box";
    node.style.left = `${Math.max(0, box[0] / width * 100)}%`;
    node.style.top = `${Math.max(0, box[1] / height * 100)}%`;
    node.style.width = `${Math.max(0, (box[2] - box[0]) / width * 100)}%`;
    node.style.height = `${Math.max(0, (box[3] - box[1]) / height * 100)}%`;
    const label = document.createElement("span");
    label.textContent = overlay.label || overlay.track_id || "Vehicle";
    node.append(label);
    elements["frame-overlay"].append(node);
  }
}

function emptyTableRow(message) {
  const row = document.createElement("tr");
  row.className = "empty-row";
  const cell = document.createElement("td");
  cell.colSpan = 4;
  cell.textContent = message;
  row.append(cell);
  return row;
}

function predictionFrom(track) {
  return track.prediction || track.vehicle_prediction || track;
}

function renderTracks(tracks) {
  elements["track-count"].textContent = String(tracks.length);
  if (tracks.length === 0) {
    elements["tracks-body"].replaceChildren(emptyTableRow("No active tracks in the current frame window."));
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const track of tracks) {
    const trackId = track.track_id || track.id;
    const prediction = predictionFrom(track);
    const deepest = prediction.model_family?.accepted ? prediction.model_family : prediction.make?.accepted ? prediction.make : prediction.body_type;
    const row = document.createElement("tr");
    row.className = "track-row";
    row.tabIndex = 0;
    row.dataset.trackId = trackId;
    row.dataset.testid = "track-row";
    row.setAttribute("aria-selected", String(trackId === state.selectedTrackId));
    row.setAttribute("aria-label", `Inspect track ${trackId}`);

    const idCell = document.createElement("td");
    idCell.className = "track-id-cell";
    const id = document.createElement("strong");
    id.textContent = trackId;
    const age = document.createElement("span");
    age.textContent = text(track.age_frames ? `${track.age_frames} frames` : track.state, "Active");
    idCell.append(id, age);
    const evidence = document.createElement("td");
    evidence.textContent = text(prediction.usable_frames ?? track.usable_frames, "0");
    const decision = document.createElement("td");
    decision.className = "decision-cell";
    const decisionName = document.createElement("strong");
    decisionName.textContent = text(deepest?.label, "Abstained");
    const decisionState = document.createElement("span");
    decisionState.textContent = humanize(prediction.decision || "pending");
    decision.append(decisionName, decisionState);
    const confidence = document.createElement("td");
    confidence.className = "confidence-cell";
    confidence.textContent = formatConfidence(deepest?.confidence);
    row.append(idCell, evidence, decision, confidence);
    row.addEventListener("click", () => selectTrack(trackId));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        void selectTrack(trackId);
      }
    });
    fragment.append(row);
  }
  elements["tracks-body"].replaceChildren(fragment);
}

async function selectTrack(trackId) {
  state.selectedTrackId = trackId;
  for (const row of elements["tracks-body"].querySelectorAll(".track-row")) {
    row.setAttribute("aria-selected", String(row.dataset.trackId === trackId));
  }
  elements["selected-track-id"].textContent = trackId;
  try {
    const payload = await api.track(trackId);
    const track = payload.track || payload;
    renderEvidence(track);
    renderHierarchy(predictionFrom(track));
  } catch (error) {
    showToast(describeError(error), "error");
  }
}

function resetEvidence() {
  elements["selected-track-id"].textContent = "No track selected";
  const empty = document.createElement("div");
  empty.className = "evidence-empty";
  empty.textContent = "Select an active track to inspect its bounded evidence ledger.";
  elements["evidence-strip"].replaceChildren(empty);
  renderHierarchy({});
}

function renderEvidence(track) {
  const evidence = track.evidence?.items || track.evidence_items || track.items || [];
  if (evidence.length === 0) {
    const empty = document.createElement("div");
    empty.className = "evidence-empty";
    empty.textContent = "No display-safe crop evidence is available for this track.";
    elements["evidence-strip"].replaceChildren(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const item of evidence) {
    const card = document.createElement("figure");
    card.className = "evidence-item";
    const imageUrl = item.crop_url || item.image_url;
    if (imageUrl) {
      const image = document.createElement("img");
      image.src = imageUrl;
      image.alt = `Privacy-safe evidence crop from frame ${text(item.frame_id)}`;
      card.append(image);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "evidence-placeholder";
      placeholder.textContent = `Frame ${text(item.frame_id)}`;
      card.append(placeholder);
    }
    const caption = document.createElement("span");
    caption.textContent = `Weight ${Number(item.fusion_weight ?? item.quality?.fusion_weight ?? 0).toFixed(2)}`;
    card.append(caption);
    fragment.append(card);
  }
  elements["evidence-strip"].replaceChildren(fragment);
}

function renderHierarchy(prediction) {
  for (const levelName of ["body_type", "make", "model_family"]) {
    const level = prediction[levelName] || {};
    const container = elements["hierarchy-panel"].querySelector(`[data-level="${levelName}"]`);
    const label = container.querySelector("strong");
    const bar = container.querySelector(".confidence-bar");
    const fill = bar.querySelector("span");
    const chip = container.querySelector(".decision-chip");
    const confidence = Number(level.confidence);
    const percentage = Number.isFinite(confidence) ? Math.round(Math.max(0, Math.min(1, confidence)) * 100) : 0;
    label.textContent = text(level.label, "Awaiting evidence");
    fill.style.width = `${percentage}%`;
    bar.setAttribute("aria-valuenow", String(percentage));
    chip.dataset.accepted = String(level.accepted === true);
    chip.dataset.state = level.accepted === false && level.label ? "abstained" : "pending";
    chip.textContent = level.accepted === true ? `Accepted ${percentage}%` : level.label ? `Abstained ${percentage}%` : "Pending";
  }

  const decision = String(prediction.decision || "");
  const acceptedModel = decision === "ACCEPT_BODY_MAKE_MODEL";
  const abstained = decision && !acceptedModel;
  elements["abstention-panel"].dataset.tone = acceptedModel ? "accepted" : "warning";
  const title = elements["abstention-panel"].querySelector("strong");
  const detail = elements["abstention-panel"].querySelector("p");
  title.textContent = acceptedModel ? "Model family accepted" : abstained ? humanize(decision) : "Decision pending";
  detail.textContent = prediction.abstention_reason || prediction.reason || (acceptedModel
    ? "The calibrated hierarchy accepted the displayed body, make, and model-family labels for this track."
    : abstained ? "CarFace reports only the deepest hierarchy level supported by current calibrated evidence."
      : "No confidence or accuracy claim is made before calibrated track evidence is available.");
}

function renderStageDetail(selection) {
  const event = selection.event;
  elements["stage-detail-title"].textContent = selection.label;
  elements["stage-detail-status"].textContent = humanize(event.status);
  elements["stage-detail-duration"].textContent = formatDuration(event.duration_ms);
  elements["stage-detail-context"].textContent = [event.frame_id != null ? `Frame ${event.frame_id}` : null, event.track_id].filter(Boolean).join(" · ") || "--";
  elements["stage-detail-message"].textContent = event.warning || event.error || event.error_message || (event.status === "pending" ? "Waiting for an inference run." : "Latest bounded stage event.");
  elements["stage-detail-data"].textContent = JSON.stringify({ input: event.input_summary || {}, output: event.output_summary || {}, error_code: event.error_code || null }, null, 2);
  document.querySelector(".stage-status-dot").dataset.status = event.status;
}

elements["source-select"].addEventListener("change", updateSourceContext);
elements["start-button"].addEventListener("click", startRun);
elements["pause-button"].addEventListener("click", () => transitionRun("pause"));
elements["resume-button"].addEventListener("click", () => transitionRun("resume"));
elements["stop-button"].addEventListener("click", () => transitionRun("stop"));
window.addEventListener("beforeunload", () => {
  stopLiveUpdates();
  if (state.currentObjectUrl) URL.revokeObjectURL(state.currentObjectUrl);
});

await Promise.all([loadStatus(), loadSources()]);
setRunState("pending");

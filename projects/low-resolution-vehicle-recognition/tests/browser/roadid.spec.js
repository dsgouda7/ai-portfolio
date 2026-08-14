import { expect, test } from "@playwright/test";

const JPEG_BASE64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EB//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EB//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EB//2Q==";

const longError = "Detector worker rejected a deliberately long diagnostic message because the configured model bundle and runtime device metadata could not be reconciled. This text must wrap without obscuring controls, changing graph dimensions, or widening the application viewport.";

async function installMockApi(page, options = {}) {
  const observed = { eventHeaders: [], frameRequests: 0, stopCalls: 0 };
  const finalSequence = options.contiguousEvents ? 2 : 3;
  const events = [
    ": heartbeat\n\n",
    `id: 1\nevent: pipeline\ndata: ${JSON.stringify({ sequence_id: 1, run_id: "run-demo", frame_id: 7, stage: "source_acquisition", status: "running", started_at: "2026-08-13T12:00:00Z", input_summary: {}, output_summary: {} })}\n\n`,
    `id: ${finalSequence}\nevent: pipeline\ndata: ${JSON.stringify({ sequence_id: finalSequence, run_id: "run-demo", frame_id: 7, stage: options.longError ? "vehicle_detection" : "source_acquisition", status: options.longError ? "failed" : "completed", duration_ms: 8.4, input_summary: { source: "replay-demo" }, output_summary: { frame_id: 7 }, error_code: options.longError ? "MODEL_RUNTIME_MISMATCH" : null, error: options.longError ? longError : null })}\n\n`,
  ].join("");

  await page.route("**/api/status", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ ready: true, device: "CPU", model_version: "demo-fixture", demo_mode: true }),
  }));
  await page.route("**/api/sources", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ sources: [{ source_id: "replay-demo", name: "Licensed demo replay", adapter_type: "replay", enabled: true, attribution: "CarFace deterministic replay fixture" }] }),
  }));
  await page.route("**/api/runs", (route) => route.fulfill({
    status: 201,
    contentType: "application/json",
    body: JSON.stringify({ run_id: "run-demo", state: "running" }),
  }));
  await page.route("**/api/runs/run-demo/events", (route) => {
    observed.eventHeaders.push(route.request().headers());
    route.fulfill({ status: 200, headers: { "content-type": "text/event-stream", "cache-control": "no-cache" }, body: observed.eventHeaders.length === 1 ? events : ": heartbeat\n\n" });
  });
  await page.route("**/api/runs/run-demo/frame?*", (route) => {
    observed.frameRequests += 1;
    if (options.delayedFrame && observed.frameRequests === 1) {
      route.fulfill({
        status: 404,
        contentType: "application/json",
        body: JSON.stringify({ error: { code: "FRAME_NOT_AVAILABLE", message: "No display-safe frame is available." } }),
      });
      return;
    }
    route.fulfill({
      status: 200,
      headers: { "content-type": "image/jpeg", "x-frame-id": "7", "x-captured-at": "2026-08-13T12:00:00Z" },
      body: Buffer.from(JPEG_BASE64, "base64"),
    });
  });
  await page.route("**/api/runs/run-demo/tracks", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ tracks: [{ track_id: "track-47", age_frames: 14, prediction: { usable_frames: 6, decision: "ACCEPT_BODY_MAKE", body_type: { label: "SUV", confidence: 0.91, accepted: true }, make: { label: "Toyota", confidence: 0.76, accepted: true }, model_family: { label: "RAV4", confidence: 0.52, accepted: false } } }] }),
  }));
  await page.route("**/api/runs/run-demo", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ run_id: "run-demo", state: options.terminalRun ? "completed" : "running" }),
  }));
  await page.route("**/api/tracks/track-47", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      track_id: "track-47",
      prediction: {
        decision: "ACCEPT_BODY_MAKE",
        usable_frames: 6,
        abstention_reason: "Model-family evidence remains below the configured demo threshold.",
        body_type: { label: "SUV", confidence: 0.91, accepted: true },
        make: { label: "Toyota", confidence: 0.76, accepted: true },
        model_family: { label: "RAV4", confidence: 0.52, accepted: false },
      },
      evidence: { items: [{ crop_id: "crop-1", frame_id: 7, fusion_weight: 0.74, crop_url: `data:image/jpeg;base64,${JPEG_BASE64}` }] },
    }),
  }));
  await page.route("**/api/runs/run-demo/pause", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ run_id: "run-demo", state: "paused" }) }));
  await page.route("**/api/runs/run-demo/resume", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ run_id: "run-demo", state: "running" }) }));
  await page.route("**/api/runs/run-demo/stop", (route) => {
    observed.stopCalls += 1;
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ run_id: "run-demo", state: "stopped" }) });
  });
  return observed;
}

async function startDemo(page, expectedState = "Running") {
  await page.goto("/");
  await page.getByTestId("source-select").selectOption("replay-demo");
  await page.getByTestId("start-button").click();
  await expect(page.getByTestId("run-state")).toHaveText(expectedState);
}

async function expectNoViewportOverflow(page) {
  const dimensions = await page.evaluate(() => ({ viewport: window.innerWidth, document: document.documentElement.scrollWidth }));
  expect(dimensions.document).toBeLessThanOrEqual(dimensions.viewport + 1);
}

function overlaps(first, second) {
  if (!first || !second) return false;
  return first.left < second.right && first.right > second.left && first.top < second.bottom && first.bottom > second.top;
}

test("loads approved sources and labels the demo profile", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/");
  await expect(page.locator(".brand-block h1")).toHaveText("CarFace");
  await expect(page.getByTestId("source-select")).toContainText("Licensed demo replay");
  await expect(page.getByTestId("readiness-badge")).toHaveText("Ready");
  await expect(page.locator("#profile-badge")).toContainText("Demo profile");
  await expect(page.getByTestId("start-button")).toBeDisabled();
  await page.getByTestId("source-select").selectOption("replay-demo");
  await expect(page.locator("#profile-badge")).toHaveText("Demo profile · non-production");
  await expect(page.getByTestId("start-button")).toBeEnabled();
});

test("starts, transitions the graph, selects a track, reconnects, and stops", async ({ page }) => {
  const observed = await installMockApi(page);
  await startDemo(page);

  await expect(page.getByTestId("pipeline-stage-source_acquisition")).toHaveAttribute("data-status", "completed");
  await expect(page.getByTestId("tracks-body").getByTestId("track-row")).toBeVisible();
  await page.getByTestId("tracks-body").getByTestId("track-row").click();
  await expect(page.getByTestId("evidence-strip").locator(".evidence-item")).toHaveCount(1);
  await expect(page.locator('[data-level="make"] strong')).toHaveText("Toyota");
  await expect(page.getByTestId("abstention-panel")).toContainText("Accept Body Make");
  await expect(page.getByTestId("frame-viewport").locator("img")).toBeVisible();
  await expect(page.locator("#connection-notice")).toContainText("Event sequence gap");

  await expect.poll(() => observed.eventHeaders.length).toBeGreaterThan(1);
  expect(observed.eventHeaders[1]["last-event-id"]).toBe("3");
  const sequenceBeforeStop = await page.locator("#event-sequence").textContent();
  await page.getByTestId("stop-button").click();
  await expect(page.getByTestId("run-state")).toHaveText("Stopped");
  await expect(page.getByTestId("stop-button")).toBeDisabled();
  expect(observed.stopCalls).toBe(1);
  await page.waitForTimeout(900);
  await expect(page.locator("#event-sequence")).toHaveText(sequenceBeforeStop);
});

test("a completed run closes its finite event stream without reconnecting", async ({ page }) => {
  const observed = await installMockApi(page, { terminalRun: true, contiguousEvents: true });
  await startDemo(page, "Completed");

  await expect(page.getByTestId("run-state")).toHaveText("Completed");
  await expect(page.locator("#trace-connection")).toHaveText("Complete");
  await expect(page.locator("#connection-notice")).toBeHidden();
  await page.waitForTimeout(1200);
  expect(observed.eventHeaders).toHaveLength(1);
});

test("a startup frame miss is retried after the run completes", async ({ page }) => {
  const observed = await installMockApi(page, { terminalRun: true, contiguousEvents: true, delayedFrame: true });
  await startDemo(page, "Completed");

  await expect(page.getByTestId("frame-viewport").locator("img")).toBeVisible();
  expect(observed.frameRequests).toBeGreaterThanOrEqual(2);
});

test("live replay completes without an event-stream reconnect warning", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#profile-badge")).toHaveText("Demo profile · non-production");
  await page.getByTestId("source-select").selectOption("replay-demo");
  await page.getByTestId("start-button").click();

  await expect(page.getByTestId("run-state")).toHaveText("Completed", { timeout: 15_000 });
  await expect(page.locator("#trace-connection")).toHaveText("Complete");
  await expect(page.locator("#connection-notice")).not.toContainText("reconnecting");
  const frame = page.getByTestId("frame-viewport").locator("img");
  await expect(frame).toBeVisible();
  expect(await frame.evaluate((image) => image.naturalWidth)).toBeGreaterThan(0);
  const vehiclePixels = await frame.evaluate((image) => {
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let count = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      if (pixels[index] > 140 && pixels[index] > pixels[index + 1] + 40 && pixels[index + 1] > pixels[index + 2]) count += 1;
    }
    return count;
  });
  expect(vehiclePixels).toBeGreaterThan(100);
  await expect(page.getByTestId("tracks-body").getByTestId("track-row")).toBeVisible();
});

test("long stage errors wrap without changing graph dimensions", async ({ page }) => {
  await installMockApi(page, { longError: true });
  await startDemo(page);
  await expect(page.getByTestId("pipeline-stage-vehicle_detection")).toHaveAttribute("data-status", "failed");
  await expect(page.locator("#stage-detail-message")).toContainText("deliberately long diagnostic");
  const bounds = await page.locator("#stage-detail-message").evaluate((node) => ({ width: node.clientWidth, scroll: node.scrollWidth }));
  expect(bounds.scroll).toBeLessThanOrEqual(bounds.width + 1);
  await expectNoViewportOverflow(page);
});

test("desktop layout has stable nonblank primary surfaces", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "Desktop geometry assertion");
  await installMockApi(page);
  await startDemo(page);
  const image = page.getByTestId("frame-viewport").locator("img");
  await expect(image).toBeVisible();
  expect(await image.evaluate((node) => node.naturalWidth)).toBeGreaterThan(0);
  await expect(page.getByTestId("pipeline-graph").locator(".pipeline-stage")).toHaveCount(10);
  const screenshot = await page.getByTestId("primary-surface").screenshot();
  expect(screenshot.byteLength).toBeGreaterThan(5000);
  const boxes = await page.evaluate(() => {
    const live = document.querySelector(".live-panel").getBoundingClientRect();
    const tracks = document.querySelector(".tracks-panel").getBoundingClientRect();
    return { live, tracks };
  });
  expect(overlaps(boxes.live, boxes.tracks)).toBe(false);
  await expectNoViewportOverflow(page);
});

test("mobile layout stacks panels and keeps controls in bounds", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "Mobile geometry assertion");
  await installMockApi(page);
  await startDemo(page);
  const geometry = await page.evaluate(() => {
    const live = document.querySelector(".live-panel").getBoundingClientRect();
    const tracks = document.querySelector(".tracks-panel").getBoundingClientRect();
    const toolbar = document.querySelector(".run-toolbar").getBoundingClientRect();
    return { liveBottom: live.bottom, tracksTop: tracks.top, toolbarLeft: toolbar.left, toolbarRight: toolbar.right, viewport: innerWidth };
  });
  expect(geometry.tracksTop).toBeGreaterThanOrEqual(geometry.liveBottom);
  expect(geometry.toolbarLeft).toBeGreaterThanOrEqual(0);
  expect(geometry.toolbarRight).toBeLessThanOrEqual(geometry.viewport + 1);
  await expect(page.getByTestId("pipeline-graph").locator(".pipeline-stage")).toHaveCount(10);
  await expectNoViewportOverflow(page);
});

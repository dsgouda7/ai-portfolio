import { expect, test } from "@playwright/test";

const JPEG_BASE64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EB//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EB//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EB//2Q==";

async function mockApi(page) {
  let jobPoll = 0;
  let trainingPoll = 0;
  const completedPipeline = [
    ["fetch", "Fetch provider metadata"],
    ["download", "Download uncached images"],
    ["preprocess", "Normalize and enhance images"],
    ["baseline-inference", "Run SpeciesNet baseline"],
    ["evaluate", "Score deployed model"],
    ["train", "Build next candidate catalog"],
    ["persist", "Persist and deploy version"],
  ].map(([id, label]) => ({ id, label, state: "completed", processed: 12, total: 12, detail: `${label} complete.` }));
  await page.route("https://unpkg.com/**", (route) => {
    if (route.request().url().endsWith("leaflet.js")) {
      return route.fulfill({ status: 200, contentType: "application/javascript", body: `
        window.L = {
          map(element) { return { element, fitBounds() {}, setView() {}, invalidateSize() {} }; },
          tileLayer() { return { addTo() { return this; } }; },
          layerGroup() { return { addTo() { return this; }, clearLayers() {} }; },
          marker() {
            const listeners = {};
            const button = document.createElement("button");
            button.type = "button"; button.className = "mock-map-marker"; button.textContent = "Observation point";
            document.getElementById("observation-map").append(button);
            button.addEventListener("click", () => listeners.click?.());
            return { addTo() { return this; }, bindTooltip() { return this; }, on(name, handler) { listeners[name] = handler; return this; } };
          },
        };
      ` });
    }
    return route.fulfill({ status: 200, body: "" });
  });
  await page.route("https://*.tile.openstreetmap.org/**", (route) => route.abort());
  await page.route("**/api/status", (route) => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ service: "WildScope", ready: true, static_model: "SpeciesNet 5.0.5", feed_count: 10, page_size: 10 }),
  }));
  await page.route("**/api/feeds", (route) => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ feeds: [
      { feed_id: "yasuni", name: "Yasuni National Park", place_id: 68650, country: "ECU", habitat: "Amazon rainforest", adaptive_model: { model_id: "adaptive-yasuni", trained_at: "2026-08-18T12:00:00Z", watermark: "2026-08-18T10:00:00Z", sample_count: 30 } },
      ...Array.from({ length: 9 }, (_, index) => ({ feed_id: `feed-${index}`, name: `Tropical Feed ${index + 2}`, place_id: 70000 + index, country: "CRI", habitat: "tropical forest", adaptive_model: null })),
    ] }),
  }));
  await page.route("**/api/feeds/yasuni/sync", (route) => route.fulfill({
    status: 202, contentType: "application/json", body: JSON.stringify({ job: { job_id: "sync-1", state: "running" } }),
  }));
  await page.route("**/api/feeds/yasuni/train", (route) => route.fulfill({
    status: 202, contentType: "application/json", body: JSON.stringify({ job: { job_id: "train-1", state: "running" } }),
  }));
  await page.route("**/api/jobs/sync-1", (route) => {
    jobPoll += 1;
    route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ job: jobPoll === 1
        ? { job_id: "sync-1", kind: "sync", state: "running", processed: 5, total: 10, details: {} }
        : { job_id: "sync-1", kind: "sync", state: "completed", processed: 10, total: 10, details: { observations: 12, duplicates_skipped: 2, static_predictions: 10 } } }),
    });
  });
  await page.route("**/api/jobs/train-1", (route) => {
    trainingPoll += 1;
    route.fulfill({
      status: 200, contentType: "application/json",
      body: JSON.stringify({ job: trainingPoll === 1
        ? { job_id: "train-1", feed_id: "yasuni", kind: "training", state: "running", processed: 5, total: 12, details: { pipeline: completedPipeline.map((stage, index) => index < 3 ? stage : { ...stage, state: index === 3 ? "running" : "pending" }), current_stage: "baseline-inference" } }
        : { job_id: "train-1", feed_id: "yasuni", kind: "training", state: "completed", processed: 12, total: 12, details: { pipeline: completedPipeline, evaluated_model_id: "adaptive-yasuni-v3", trained_model_id: "adaptive-yasuni-v4", new_samples: 12, training_samples: 12, downloaded: 12, duration_seconds: 8.4, baseline_accuracy: 0.7, baseline_mean_confidence: 0.765, baseline_coverage: 1, deployed_accuracy: 0.79, deployed_mean_confidence: 0.855, deployed_coverage: 1, target_count: 2, training_agreement: 0.92, trained_at: "2026-08-18T13:00:00Z" } } }),
    });
  });
  const items = Array.from({ length: 10 }, (_, index) => ({
    photo_id: index + 1,
    image_url: `data:image/jpeg;base64,${JPEG_BASE64}`,
    common_name: index % 2 ? "Ocelot" : "Jaguar",
    scientific_name: index % 2 ? "Leopardus pardalis" : "Panthera onca",
    created_at: "2026-08-18T10:00:00Z",
    quality_grade: "research",
    license_code: "cc-by",
    attribution: "Observer",
    latitude: -0.7002,
    longitude: -78.3001,
    coordinates_obscured: false,
    static_label: index % 2 ? "Leopardus pardalis" : "Panthera onca",
    static_confidence: 0.72 + index / 100,
    static_identification: {
      scientific_name: null,
      common_name: index % 2 ? "ocelot" : "jaguar",
      source_label: index % 2 ? "mammalia;ocelot" : "mammalia;jaguar",
      ambiguous: false,
    },
    adaptive_label: index === 1 ? "Leopardus pardalis" : "unidentified",
    adaptive_raw_label: index % 2 ? "Leopardus pardalis" : "Panthera onca",
    adaptive_confidence: null,
    adaptive_margin: index === 1 ? 0.08 : 0.02,
    adaptive_identification: {
      scientific_name: index === 1 ? "Leopardus pardalis" : null,
      common_name: index === 1 ? "Ocelot" : "Unidentified",
      source_label: "hf-hub:imageomics/bioclip",
      candidate_count: 2,
      candidate_scientific_names: ["Leopardus pardalis", "Panthera onca"],
      ambiguous: index !== 1,
      abstained: index !== 1,
      decision_metric: "top-1 minus top-2 cosine similarity",
      decision_margin: index === 1 ? 0.08 : 0.02,
      margin_threshold: 0.075,
    },
    obtained_identification: {
      source: "iNaturalist community identification",
      common_name: index % 2 ? "Ocelot" : "Jaguar",
      scientific_name: index % 2 ? "Leopardus pardalis" : "Panthera onca",
      quality_grade: "research",
      research_grade: true,
    },
    static_match: true,
    adaptive_match: index === 1,
  }));
  await page.route("**/api/feeds/yasuni/frames?page=1", (route) => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ page: 1, pages: 2, total: 12, items }),
  }));
  await page.route("**/api/feeds/yasuni/locations", (route) => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ locations: [{ anchor_photo_id: 1, latitude: -0.7002, longitude: -78.3001, positional_accuracy: 20, coordinates_obscured: 0, photo_count: 10, common_name: "Jaguar" }] }),
  }));
  await page.route("**/api/feeds/yasuni/locations/1/frames", (route) => route.fulfill({
    status: 200, contentType: "application/json", body: JSON.stringify({ items: items.slice(0, 3) }),
  }));
  await page.route("**/api/frames/1", (route) => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ ...items[0], stages: [
      { id: "source", name: "Original capture", processor: "iNaturalist original asset", image_url: `data:image/jpeg;base64,${JPEG_BASE64}`, dimensions: [1600, 1000] },
      { id: "normalized", name: "Clean and normalize", processor: "Pillow EXIF transpose + RGB normalization", image_url: `data:image/jpeg;base64,${JPEG_BASE64}`, dimensions: [1600, 1000] },
      { id: "enhanced", name: "Model input", processor: "original-resolution-passthrough", image_url: `data:image/jpeg;base64,${JPEG_BASE64}`, dimensions: [1600, 1000] },
      { id: "classification", name: "Identify wildlife", processor: "SpeciesNet 5.0.5", obtained: items[0].obtained_identification, static: { label: "Panthera onca", confidence: 0.72, matches_obtained: true, identification: items[0].static_identification }, adaptive: { label: "unidentified", confidence: null, margin: 0.02, matches_obtained: false, identification: items[0].adaptive_identification } },
    ] }),
  }));
  await page.route("**/api/feeds/yasuni/training", (route) => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({
      model: { model_id: "adaptive-yasuni-v3", trained_at: "2026-08-18T10:00:00Z", watermark: "2026-08-18T10:00:00Z", sample_count: 30, training_samples: 30, protocol_version: "test-then-train-v3-bioclip-selective" },
      live_batch: {
        status: "ready", evaluated_model_id: "adaptive-yasuni-v3",
        window_from: "2026-08-18T10:00:00Z", window_to: "2026-08-18T12:00:00Z",
        eligible_samples: 12,
        baseline: { samples: 12, correct: 8, errors: 4, accuracy: 0.7, coverage: 1, mean_confidence: 0.765, target_count: 2 },
        deployed: { samples: 12, correct: 9, errors: 3, accuracy: 0.79, coverage: 1, mean_confidence: 0.855, target_count: 2 },
        samples: items.map((item) => ({
          photo_id: item.photo_id, observation_id: item.photo_id,
          created_at: item.created_at,
          obtained_scientific_name: item.scientific_name,
          obtained_common_name: item.common_name, quality_grade: "research",
          baseline_label: item.static_label, baseline_confidence: item.static_confidence,
          deployed_label: item.adaptive_label, deployed_confidence: item.adaptive_confidence,
          deployed_margin: item.adaptive_margin,
          deployed_model_version: "bioclip-vit-b16-selective-margin-0.075",
        })),
      },
      baseline_model: { name: "SpeciesNet", engine_version: "speciesnet-5.0.5", prediction_versions: ["4.0.3a"] },
      dataset: { total_observations: 16, cached_images: 14, licensed_images: 14, excluded_unlicensed: 2, research_grade: 12, excluded_not_research: 4, eligible_labels: 12, baseline_predictions: 14, baseline_prediction_versions: ["4.0.3a"], target_distribution: { "Panthera onca": 6, "Leopardus pardalis": 6 } },
      confidence: { sample_count: 10, baseline_mean_confidence: 0.765, adaptive_mean_confidence: 0.855, confidence_delta: 0.09 },
      runs: [{ run_id: "train-0", started_at: "2026-08-17T11:59:51Z", finished_at: "2026-08-17T12:00:00Z", details: { evaluated_model_id: "adaptive-yasuni-v2", trained_model_id: "adaptive-yasuni-v3", training_samples: 10, duration_seconds: 9, baseline_accuracy: 0.68, deployed_accuracy: 0.75, training_agreement: 0.9, watermark: "2026-08-18T10:00:00Z" } }],
    }),
  }));
  await page.route("**/api/feeds/feed-0/training", (route) => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({
      model: null,
      baseline_model: { name: "SpeciesNet", engine_version: "speciesnet-5.0.5", prediction_versions: ["4.0.3a"] },
      live_batch: { status: "awaiting-data", evaluated_model_id: "SpeciesNet baseline", window_from: null, window_to: null, eligible_samples: 0, baseline: { samples: 0, correct: 0, errors: 0, accuracy: null, coverage: null, mean_confidence: null, target_count: 0 }, deployed: { samples: 0, correct: null, errors: null, accuracy: null, coverage: null, mean_confidence: null, target_count: 0 }, samples: [] },
      dataset: { total_observations: 4, cached_images: 4, licensed_images: 3, excluded_unlicensed: 1, research_grade: 2, excluded_not_research: 2, eligible_labels: 2, baseline_predictions: 4, baseline_prediction_versions: ["4.0.3a"], target_distribution: { "Puma concolor": 2 } },
      confidence: {}, runs: [],
    }),
  }));
}

async function noOverflow(page) {
  const sizes = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
  expect(sizes.page).toBeLessThanOrEqual(sizes.viewport + 1);
}

test("shows ten tropical feeds and pages ten animal frames", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");

  await expect(page.locator("h1")).toHaveText("WildScope");
  await expect(page.getByTestId("feed-list").locator("button")).toHaveCount(10);
  await expect(page.locator('.feed-button[aria-current="true"]')).toHaveAttribute("data-feed-id", "yasuni");
  await expect(page.getByTestId("frame-grid").locator(".observation-card")).toHaveCount(10);
  await expect(page.locator("#sync-title")).toHaveText("Cached observations ready");
  await page.getByTestId("feed-list").locator('[data-feed-id="yasuni"]').click();

  await expect(page.getByTestId("frame-grid").locator(".observation-card")).toHaveCount(10);
  await expect(page.locator("#page-label")).toHaveText("Page 1 of 2 · 12 frames");
  await expect(page.locator("#confidence-chart")).toBeVisible();
  await expect(page.getByRole("link", { name: "Inspect training batch" })).toBeVisible();
  await expect(page.locator("#observation-map")).toBeVisible();
  await expect(page.locator("#live-batch-samples")).toHaveText("12");
  await expect(page.locator("#live-model-version")).toHaveText("adaptive-yasuni-v3");
  await expect(page.locator("#live-accuracy-delta")).toHaveText("+9.0 pts");
  const first = page.getByTestId("frame-grid").locator(".observation-card").first();
  await expect(first.locator(".obtained-identification")).toContainText("Jaguar");
  await expect(first.locator(".prediction")).toContainText("Unidentified");
  await expect(first.locator(".match-state")).toHaveText("Differs from obtained ID");
  expect(await firstImageFit(page)).toBe("contain");
  await page.locator(".mock-map-marker").click();
  await expect(page.locator(".location-strip button")).toHaveCount(3);
});

async function firstImageFit(page) {
  return page.getByTestId("frame-grid").locator(".observation-card img").first().evaluate((element) => getComputedStyle(element).objectFit);
}

test("switches between baseline confidence and selective BioCLIP margin", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByTestId("feed-list").locator('[data-feed-id="yasuni"]').click();
  const first = page.getByTestId("frame-grid").locator(".observation-card").first();
  await expect(page.locator('[data-model="adaptive"]')).toHaveAttribute("aria-pressed", "true");
  await expect(first.locator(".prediction b")).toHaveText("--");

  await page.locator('[data-model="static"]').click();

  await expect(first.locator(".prediction b")).toHaveText("72%");
  await expect(page.locator('[data-model="static"]')).toHaveAttribute("aria-pressed", "true");

  await page.locator('[data-model="adaptive"]').click();

  await expect(first.locator(".prediction b")).toHaveText("--");
  await expect(first.locator(".prediction strong")).toHaveText("Unidentified");
  await expect(first.locator(".ambiguity-state")).toHaveText("No species label emitted");
  await expect(page.locator("#adaptive-model")).toContainText("Selective model deployed");
  await expect(page.getByTestId("frame-grid").locator(".observation-card").nth(1).locator(".prediction b")).toHaveText("Margin 0.080");
});

test("mobile layout has no horizontal overflow", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "Mobile geometry assertion");
  await mockApi(page);
  await page.goto("/");
  await page.getByTestId("feed-list").locator('[data-feed-id="yasuni"]').click();
  await noOverflow(page);
});

test("opens side-by-side frame pipeline detail", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByTestId("feed-list").locator('[data-feed-id="yasuni"]').click();
  await page.getByTestId("frame-grid").locator(".detail-command").first().click();

  await expect(page.locator("#frame-dialog")).toBeVisible();
  await expect(page.locator(".pipeline-stage")).toHaveCount(4);
  await expect(page.locator("#frame-dialog")).toContainText("SpeciesNet 5.0.5");
  await expect(page.locator("#frame-dialog")).toContainText("Obtained · iNaturalist");
  await expect(page.locator("#frame-dialog")).toContainText("Abstained on hf-hub:imageomics/bioclip");
  await expect(page.locator("#frame-dialog")).toContainText("Margin 0.020");
});

test("training portal runs watermark-based training on demand", async ({ page }) => {
  await mockApi(page);
  await page.goto("/training");
  await expect(page.locator("#training-feeds input[type=checkbox]")).toHaveCount(10);
  await expect(page.locator('#training-feeds input[value="yasuni"]')).toBeChecked();
  await expect(page.locator("#pending-samples")).toHaveText("12");
  await expect(page.locator("#selected-feed-count")).toHaveText("1");
  await expect(page.locator("#baseline-model-name")).toContainText("SpeciesNet");
  await expect(page.locator("#baseline-model-versions")).toHaveText("4.0.3a");
  await expect(page.locator("#dataset-total")).toHaveText("16");
  await expect(page.locator("#dataset-eligible")).toHaveText("12");
  await expect(page.locator("#baseline-counts")).toHaveText("8 / 4");
  await expect(page.locator("#baseline-coverage")).toHaveText("100.0%");
  await expect(page.locator("#training-pipeline li")).toHaveCount(7);

  await page.locator('#training-feeds input[value="feed-0"]').check();
  await expect(page.locator("#selected-feed-count")).toHaveText("2");
  await expect(page.locator("#dataset-total")).toHaveText("20");
  await expect(page.locator("#pending-targets")).toHaveText("3");

  await expect(page.locator("#pending-samples")).toHaveText("12");
  await expect(page.locator("#pending-baseline")).toHaveText("66.7%");
  await expect(page.locator("#pending-deployed")).toHaveText("75.0%");
  await expect(page.locator("#batch-ledger tr")).toHaveCount(10);
  await expect(page.locator("#batch-ledger tr").first()).toContainText("Margin 0.020");
  await expect(page.locator("#run-training")).toHaveText("Train 2 selected feeds on 12 labels");
  await page.locator("#run-training").click();
  await expect(page.locator("#training-status-title")).toHaveText("Selected training jobs complete");
  await expect(page.locator("#training-pipeline li[data-state=completed]")).toHaveCount(7);
  await expect(page.locator("#training-ledger tr")).toHaveCount(1);
});

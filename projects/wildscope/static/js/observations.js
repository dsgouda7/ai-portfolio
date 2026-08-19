const state = {
  feeds: [],
  feedId: null,
  model: "static",
  page: 1,
  pages: 1,
  jobId: null,
  jobTimer: null,
  items: [],
  map: null,
  markerLayer: null,
};

const elements = Object.fromEntries([
  "service-state", "static-model", "adaptive-model", "feed-list", "feed-habitat",
  "workspace-title", "refresh-button", "sync-status", "sync-title",
  "sync-detail", "sync-progress", "frame-grid", "previous-page", "next-page",
  "page-label", "confidence-chart", "toast-region", "observation-map", "map-summary",
  "location-results", "live-batch-samples", "live-baseline-accuracy",
  "live-deployed-accuracy", "live-accuracy-delta", "live-model-version",
  "live-batch-window", "live-batch-status",
  "frame-dialog", "frame-dialog-close", "frame-dialog-title", "frame-dialog-meta",
  "pipeline-stages",
].map((id) => [id, document.getElementById(id)]));

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message || `Request failed (${response.status})`);
  return payload;
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  elements["toast-region"].append(toast);
  setTimeout(() => toast.remove(), 6000);
}

async function initialize() {
  try {
    const [status, feedPayload] = await Promise.all([request("/api/status"), request("/api/feeds")]);
    state.feeds = feedPayload.feeds || [];
    elements["service-state"].textContent = status.ready ? "Ready" : "Unavailable";
    elements["service-state"].dataset.tone = status.ready ? "ready" : "failed";
    elements["static-model"].textContent = `Static ${status.static_model}`;
    renderFeeds();
    const initialFeed = state.feeds.find((feed) => feed.adaptive_model) || state.feeds[0];
    if (initialFeed) await selectFeed(initialFeed.feed_id, { refresh: false });
  } catch (error) {
    elements["service-state"].textContent = "Unavailable";
    elements["service-state"].dataset.tone = "failed";
    showToast(error.message);
  }
}

function renderFeeds() {
  const fragment = document.createDocumentFragment();
  for (const feed of state.feeds) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "feed-button";
    button.dataset.feedId = feed.feed_id;
    button.setAttribute("aria-current", String(feed.feed_id === state.feedId));
    const name = document.createElement("strong");
    name.textContent = feed.name;
    const habitat = document.createElement("small");
    habitat.textContent = feed.habitat;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = feed.adaptive_model ? "trained" : "static";
    button.append(name, habitat, count);
    button.addEventListener("click", () => selectFeed(feed.feed_id));
    fragment.append(button);
  }
  elements["feed-list"].replaceChildren(fragment);
}

async function selectFeed(feedId, { refresh = true } = {}) {
  if (state.jobTimer) clearTimeout(state.jobTimer);
  state.feedId = feedId;
  state.page = 1;
  const feed = state.feeds.find((item) => item.feed_id === feedId);
  setModelMode(feed.adaptive_model ? "adaptive" : "static");
  renderFeeds();
  elements["workspace-title"].textContent = `${feed.name} · last 24 hours`;
  elements["feed-habitat"].textContent = `${feed.country} · ${feed.habitat} · insects, spiders and reptiles hidden`;
  elements["adaptive-model"].textContent = feed.adaptive_model
    ? `Selective model deployed ${formatTime(feed.adaptive_model.trained_at)}`
    : "Deployed species model not trained";
  elements["refresh-button"].disabled = false;
  elements["location-results"].replaceChildren(document.createTextNode("Loading public observation coordinates."));
  if (refresh) {
    await startSync();
    return;
  }
  setBusy(false, "Cached observations ready", "Showing the latest locally analyzed frames. Refresh when you want to query the provider.");
  await Promise.all([loadFrames(), loadLocations(), loadTrainingDashboard()]);
}

async function startSync() {
  if (!state.feedId) return;
  setBusy(true, "Syncing observation metadata", "Fetching the selected feed's last 24 hours.");
  try {
    const payload = await request(`/api/feeds/${encodeURIComponent(state.feedId)}/sync`, {
      method: "POST",
      body: JSON.stringify({ hours: 24 }),
    });
    state.jobId = payload.job.job_id;
    pollJob();
  } catch (error) {
    setBusy(false, "Sync did not start", error.message);
    showToast(error.message);
  }
}

async function pollJob() {
  try {
    const payload = await request(`/api/jobs/${encodeURIComponent(state.jobId)}`);
    const job = payload.job;
    const percent = job.total ? Math.round(job.processed / job.total * 100) : 5;
    elements["sync-progress"].value = Math.max(0, Math.min(100, percent));
    elements["sync-title"].textContent = job.kind === "training"
      ? "Testing then training pending batch" : "Analyzing recent observations";
    elements["sync-detail"].textContent = `${job.processed} / ${job.total || "?"} · ${job.state}`;
    if (["pending", "running"].includes(job.state)) {
      state.jobTimer = setTimeout(pollJob, 1200);
      return;
    }
    elements["sync-progress"].value = job.state === "completed" ? 100 : 0;
    if (job.state === "failed") {
      setBusy(false, "Job failed", job.error || "Unknown worker error");
      showToast(job.error || "Job failed");
      return;
    }
    setBusy(false, "Analysis complete", summarizeJob(job));
    await Promise.all([loadFrames(), reloadFeeds(), loadLocations(), loadTrainingDashboard()]);
  } catch (error) {
    state.jobTimer = setTimeout(pollJob, 1600);
  }
}

function summarizeJob(job) {
  const details = job.details || {};
  if (job.kind === "training") {
    return `${details.new_samples || 0} new samples · ${details.downloaded || 0} originals downloaded · ${formatDuration(details.duration_seconds)} · trained ${formatTime(details.trained_at)}`;
  }
  return `${details.observations || 0} observations · ${details.duplicates_skipped || 0} duplicates skipped · ${details.static_predictions || 0} model predictions`;
}

async function reloadFeeds() {
  const payload = await request("/api/feeds");
  state.feeds = payload.feeds || [];
  const feed = state.feeds.find((item) => item.feed_id === state.feedId);
  if (feed?.adaptive_model) {
    elements["adaptive-model"].textContent = `Selective model deployed ${formatTime(feed.adaptive_model.trained_at)}`;
  }
  renderFeeds();
}

async function loadFrames() {
  if (!state.feedId) return;
  try {
    const payload = await request(`/api/feeds/${encodeURIComponent(state.feedId)}/frames?page=${state.page}`);
    state.items = payload.items || [];
    state.pages = payload.pages || 1;
    renderFrames();
    drawChart();
    elements["page-label"].textContent = `Page ${payload.page} of ${state.pages} · ${payload.total} frames`;
    elements["previous-page"].disabled = state.page <= 1;
    elements["next-page"].disabled = state.page >= state.pages;
  } catch (error) {
    showToast(error.message);
  }
}

function renderFrames() {
  if (!state.items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No processable animal observations were returned for this feed in the last 24 hours.";
    elements["frame-grid"].replaceChildren(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const item of state.items) {
    const card = document.createElement("article");
    card.className = "observation-card";
    const image = document.createElement("img");
    image.src = item.image_url;
    image.alt = item.common_name || item.scientific_name;
    image.loading = "lazy";
    const body = document.createElement("div");
    body.className = "card-body";
    const heading = document.createElement("div");
    heading.className = "card-heading";
    const title = document.createElement("strong");
    title.textContent = `Observation ${item.observation_id}`;
    const time = document.createElement("span");
    time.textContent = `${formatTime(item.created_at)} · ${item.quality_grade}`;
    heading.append(title, time);
    const prediction = predictionFor(item);
    const result = document.createElement("div");
    result.className = "prediction";
    const model = document.createElement("span");
    model.textContent = state.model === "adaptive" ? "Selective BioCLIP" : "Static";
    const label = document.createElement("strong");
    label.textContent = prediction.identification?.common_name
      || prediction.identification?.scientific_name
      || displayPredictionLabel(prediction.label)
      || "Pending model analysis";
    const scientific = document.createElement("small");
    scientific.className = "generated-scientific";
    scientific.textContent = prediction.identification?.scientific_name
      || `Raw model label · ${displayPredictionLabel(prediction.label) || "unavailable"}`;
    scientific.hidden = Boolean(prediction.identification?.abstained);
    const confidence = document.createElement("b");
    confidence.textContent = prediction.margin == null
      ? prediction.confidence == null ? "--" : `${Math.round(prediction.confidence * 100)}%`
      : `Margin ${Number(prediction.margin).toFixed(3)}`;
    const match = document.createElement("small");
    match.className = "match-state";
    match.dataset.match = String(Boolean(prediction.matchesObtained));
    match.textContent = prediction.matchesObtained ? "Matches obtained ID" : "Differs from obtained ID";
    const raw = document.createElement("small");
    raw.className = "raw-model-label";
    raw.textContent = `Source · ${displayPredictionLabel(prediction.identification?.source_label || prediction.label) || "unavailable"}`;
    raw.hidden = Boolean(prediction.identification?.abstained);
    const ambiguity = document.createElement("small");
    ambiguity.className = "ambiguity-state";
    ambiguity.hidden = !prediction.identification?.ambiguous;
    ambiguity.textContent = prediction.identification?.ambiguous
      ? "No species label emitted"
      : "";
    result.append(model, label, scientific, confidence, raw, match, ambiguity);
    const comparison = document.createElement("div");
    comparison.className = "identification-comparison";
    comparison.append(obtainedIdentificationBlock(item), result);
    const license = document.createElement("div");
    license.className = "license";
    license.textContent = item.license_code
      ? `${item.license_code} · ${item.attribution || "iNaturalist contributor"}`
      : "License not supplied · excluded from supervised training";
    const details = document.createElement("button");
    details.type = "button";
    details.className = "detail-command";
    details.textContent = "View pipeline";
    details.addEventListener("click", () => openFrameDetail(item.photo_id));
    body.append(heading, comparison, license, details);
    card.append(image, body);
    fragment.append(card);
  }
  elements["frame-grid"].replaceChildren(fragment);
}

function obtainedIdentificationBlock(item) {
  const target = item.obtained_identification || {};
  const block = document.createElement("div");
  block.className = "obtained-identification";
  const source = document.createElement("span");
  source.textContent = "Obtained · iNaturalist";
  const label = document.createElement("strong");
  label.textContent = target.common_name || target.scientific_name || "Unidentified";
  const detail = document.createElement("small");
  detail.textContent = [
    target.scientific_name,
    target.quality_grade ? target.quality_grade.replaceAll("_", " ") : null,
  ].filter(Boolean).join(" · ");
  block.append(source, label, detail);
  return block;
}

async function loadLocations() {
  if (!state.feedId) return;
  try {
    const payload = await request(`/api/feeds/${encodeURIComponent(state.feedId)}/locations`);
    renderMap(payload.locations || []);
  } catch (error) {
    elements["map-summary"].textContent = "Map data unavailable";
    showToast(error.message);
  }
}

function ensureMap() {
  if (state.map || !window.L) return state.map;
  state.map = window.L.map(elements["observation-map"], { zoomControl: true });
  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(state.map);
  state.markerLayer = window.L.layerGroup().addTo(state.map);
  return state.map;
}

function renderMap(locations) {
  const map = ensureMap();
  if (!map) {
    elements["map-summary"].textContent = "Map library unavailable";
    return;
  }
  state.markerLayer.clearLayers();
  const bounds = [];
  for (const location of locations) {
    const point = [Number(location.latitude), Number(location.longitude)];
    if (!point.every(Number.isFinite)) continue;
    bounds.push(point);
    const marker = window.L.marker(point).addTo(state.markerLayer);
    marker.bindTooltip(`${location.common_name || "Wildlife observation"} · ${location.photo_count} capture${location.photo_count === 1 ? "" : "s"}`);
    marker.on("click", () => loadLocationFrames(location));
  }
  if (bounds.length) map.fitBounds(bounds, { padding: [28, 28], maxZoom: 15 });
  else map.setView([0, 0], 2);
  setTimeout(() => map.invalidateSize(), 0);
  const obscured = locations.filter((item) => item.coordinates_obscured).length;
  elements["map-summary"].textContent = `${locations.length} public point${locations.length === 1 ? "" : "s"}${obscured ? ` · ${obscured} obscured by source` : ""}`;
}

async function loadLocationFrames(location) {
  try {
    const payload = await request(`/api/feeds/${encodeURIComponent(state.feedId)}/locations/${location.anchor_photo_id}/frames`);
    const heading = document.createElement("div");
    heading.className = "location-heading";
    heading.textContent = `${formatCoordinate(location.latitude)}, ${formatCoordinate(location.longitude)} · ${payload.items.length} capture${payload.items.length === 1 ? "" : "s"}`;
    const strip = document.createElement("div");
    strip.className = "location-strip";
    for (const item of payload.items) {
      const button = document.createElement("button");
      button.type = "button";
      button.title = item.common_name || item.scientific_name;
      const image = document.createElement("img");
      image.src = item.image_url;
      image.alt = item.common_name || item.scientific_name;
      button.append(image);
      button.addEventListener("click", () => openFrameDetail(item.photo_id));
      strip.append(button);
    }
    elements["location-results"].replaceChildren(heading, strip);
  } catch (error) {
    showToast(error.message);
  }
}

async function openFrameDetail(photoId) {
  try {
    const item = await request(`/api/frames/${photoId}`);
    elements["frame-dialog-title"].textContent = item.common_name || item.scientific_name;
    const coordinate = item.latitude == null
      ? "Coordinates unavailable"
      : `${formatCoordinate(item.latitude)}, ${formatCoordinate(item.longitude)}${item.coordinates_obscured ? " · source-obscured" : ""}`;
    elements["frame-dialog-meta"].textContent = `${coordinate} · ${formatTime(item.created_at)} · ${item.license_code || "unlicensed"}`;
    const fragment = document.createDocumentFragment();
    for (const stage of item.stages || []) fragment.append(renderPipelineStage(stage));
    elements["pipeline-stages"].replaceChildren(fragment);
    elements["frame-dialog"].showModal();
  } catch (error) {
    showToast(error.message);
  }
}

function renderPipelineStage(stage) {
  const article = document.createElement("article");
  article.className = "pipeline-stage";
  const heading = document.createElement("header");
  const name = document.createElement("strong");
  name.textContent = stage.name;
  const processor = document.createElement("span");
  processor.textContent = stage.processor;
  heading.append(name, processor);
  article.append(heading);
  if (stage.image_url) {
    const image = document.createElement("img");
    image.src = stage.image_url;
    image.alt = `${stage.name} output`;
    article.append(image);
    const dimensions = document.createElement("small");
    dimensions.textContent = formatDimensions(stage.dimensions);
    article.append(dimensions);
  }
  if (stage.static) {
    article.append(
      obtainedPredictionRow(stage.obtained),
      predictionRow("SpeciesNet baseline", stage.static),
      predictionRow("Latest trained", stage.adaptive),
    );
  }
  return article;
}

function obtainedPredictionRow(obtained) {
  const row = document.createElement("div");
  row.className = "stage-prediction obtained";
  const source = document.createElement("span");
  source.textContent = "Obtained · iNaturalist";
  const label = document.createElement("strong");
  label.textContent = obtained?.common_name || obtained?.scientific_name || "Unavailable";
  const quality = document.createElement("b");
  quality.textContent = obtained?.quality_grade?.replaceAll("_", " ") || "ungraded";
  row.append(source, label, quality);
  return row;
}

function predictionRow(name, prediction) {
  const row = document.createElement("div");
  row.className = "stage-prediction";
  const identification = prediction?.identification || {};
  const label = identification.common_name
    || identification.scientific_name
    || displayPredictionLabel(prediction?.label)
    || "Unavailable";
  const confidence = prediction?.margin == null
    ? prediction?.confidence == null ? "--" : formatPercent(prediction.confidence)
    : `Margin ${Number(prediction.margin).toFixed(3)}`;
  row.innerHTML = `<span>${name}</span><strong></strong><b>${confidence}</b>`;
  row.querySelector("strong").textContent = label;
  row.dataset.match = String(Boolean(prediction?.matches_obtained));
  const detail = document.createElement("small");
  detail.textContent = identification.scientific_name
    || `Raw model label · ${displayPredictionLabel(prediction?.label) || "unavailable"}`;
  row.append(detail);
  if (identification.ambiguous) {
    const warning = document.createElement("small");
    warning.className = "ambiguity-state";
    warning.textContent = `Abstained on ${identification.source_label} · ${identification.candidate_count} possible species`;
    row.append(warning);
    const candidates = document.createElement("small");
    candidates.textContent = `Candidates · ${(identification.candidate_scientific_names || []).join(", ")}`;
    row.append(candidates);
  }
  return row;
}

async function loadTrainingDashboard() {
  if (!state.feedId) return;
  try {
    const dashboard = await request(`/api/feeds/${encodeURIComponent(state.feedId)}/training`);
    renderTrainingMetrics(dashboard);
  } catch (error) {
    showToast(error.message);
  }
}

function renderTrainingMetrics(dashboard) {
  const batch = dashboard.live_batch || {};
  const baseline = batch.baseline || {};
  const deployed = batch.deployed || {};
  const delta = deployed.accuracy != null && baseline.accuracy != null
    && Number.isFinite(Number(deployed.accuracy)) && Number.isFinite(Number(baseline.accuracy))
    ? Number(deployed.accuracy) - Number(baseline.accuracy)
    : null;
  elements["live-batch-samples"].textContent = String(batch.eligible_samples || 0);
  elements["live-baseline-accuracy"].textContent = formatPercent(baseline.accuracy);
  elements["live-deployed-accuracy"].textContent = formatPercent(deployed.accuracy);
  elements["live-accuracy-delta"].textContent = formatDelta(delta);
  elements["live-accuracy-delta"].dataset.tone = Number(delta) >= 0 ? "positive" : "negative";
  elements["live-model-version"].textContent = batch.evaluated_model_id || "SpeciesNet baseline";
  elements["live-batch-window"].textContent = batch.window_to
    ? `${batch.window_from ? formatTime(batch.window_from) : "First model"} → ${formatTime(batch.window_to)}`
    : "Awaiting observations";
  elements["live-batch-status"].textContent = batch.eligible_samples
    ? batch.bootstrap_migration
      ? `${batch.eligible_samples} cached research-grade labels are ready for a one-time lifecycle bootstrap.`
      : `${batch.eligible_samples} research-grade labels are scored and ready for the next model version.`
    : "No unseen research-grade labels have arrived since this model version.";
}

function predictionFor(item) {
  if (state.model === "adaptive" && item.adaptive_label) {
    return { label: item.adaptive_label, confidence: item.adaptive_confidence, margin: item.adaptive_identification?.abstained ? null : item.adaptive_margin, matchesObtained: item.adaptive_match, identification: item.adaptive_identification };
  }
  return { label: item.static_label, confidence: item.static_confidence, matchesObtained: item.static_match, identification: item.static_identification };
}

function displayPredictionLabel(value) {
  if (!value) return null;
  const parts = String(value).split(";").map((part) => part.trim()).filter(Boolean);
  if (parts.length > 1) return parts.at(-1);
  return String(value);
}

function drawChart() {
  const canvas = elements["confidence-chart"];
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfcf8";
  context.fillRect(0, 0, width, height);
  const margin = { left: 52, right: 20, top: 24, bottom: 42 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  context.strokeStyle = "#cbd3cb";
  context.fillStyle = "#5f6b63";
  context.font = "12px Aptos";
  for (let step = 0; step <= 4; step += 1) {
    const y = margin.top + plotHeight - step / 4 * plotHeight;
    context.beginPath(); context.moveTo(margin.left, y); context.lineTo(width - margin.right, y); context.stroke();
    context.fillText(`${step * 25}%`, 8, y + 4);
  }
  const slot = plotWidth / Math.max(1, state.items.length);
  state.items.forEach((item, index) => {
    const x = margin.left + index * slot + slot * .18;
    drawBar(context, x, item.static_confidence, plotHeight, margin.top, "#185b3b", slot * .25);
    context.fillStyle = "#5f6b63";
    context.fillText(String(index + 1), x + slot * .18, height - 16);
  });
  context.fillStyle = "#185b3b"; context.fillRect(width - 220, 12, 12, 12);
  context.fillStyle = "#16201a"; context.fillText("Static", width - 202, 22);
}

function drawBar(context, x, value, plotHeight, top, color, width) {
  const confidence = Number(value);
  if (!Number.isFinite(confidence)) return;
  const barHeight = Math.max(0, Math.min(1, confidence)) * plotHeight;
  context.fillStyle = color;
  context.fillRect(x, top + plotHeight - barHeight, Math.max(3, width), barHeight);
}

function setBusy(busy, title, detail) {
  elements["sync-title"].textContent = title;
  elements["sync-detail"].textContent = detail;
  elements["refresh-button"].disabled = busy || !state.feedId;
  for (const button of elements["feed-list"].querySelectorAll("button")) button.disabled = busy;
}

function formatTime(value) {
  if (!value) return "time unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function formatPercent(value) {
  if (value == null || value === "") return "--";
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "--";
}

function formatDelta(value) {
  if (value == null || value === "") return "--";
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  const points = number * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(1)} pts`;
}

function formatDuration(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function formatCoordinate(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(5) : "--";
}

function formatDimensions(dimensions) {
  if (!Array.isArray(dimensions) || dimensions.some((value) => !Number.isFinite(Number(value)))) {
    return "Dimensions unavailable";
  }
  return `${dimensions[0]} × ${dimensions[1]} px`;
}

for (const button of document.querySelectorAll("[data-model]")) {
  button.addEventListener("click", () => {
    setModelMode(button.dataset.model);
  });
}

function setModelMode(model) {
  state.model = model;
  for (const peer of document.querySelectorAll("[data-model]")) {
    peer.setAttribute("aria-pressed", String(peer.dataset.model === model));
  }
  renderFrames();
}
elements["refresh-button"].addEventListener("click", startSync);
elements["previous-page"].addEventListener("click", () => { state.page -= 1; loadFrames(); });
elements["next-page"].addEventListener("click", () => { state.page += 1; loadFrames(); });
elements["frame-dialog-close"].addEventListener("click", () => elements["frame-dialog"].close());
elements["frame-dialog"].addEventListener("click", (event) => {
  if (event.target === elements["frame-dialog"]) elements["frame-dialog"].close();
});

await initialize();

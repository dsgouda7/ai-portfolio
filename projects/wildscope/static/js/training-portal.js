const PIPELINE_STAGES = [
  ["fetch", "Fetch provider metadata"],
  ["download", "Download uncached images"],
  ["preprocess", "Normalize and enhance images"],
  ["baseline-inference", "Run SpeciesNet baseline"],
  ["evaluate", "Score deployed model"],
  ["train", "Build next candidate catalog"],
  ["persist", "Persist and deploy version"],
];

const state = {
  feeds: [],
  selectedFeedIds: new Set(),
  dashboards: new Map(),
  busy: false,
};

const elements = Object.fromEntries([
  "training-feeds", "run-training", "training-status-title", "training-status-detail",
  "training-progress", "selected-feed-count", "pending-samples", "pending-targets",
  "pending-baseline", "baseline-counts", "pending-deployed", "deployed-counts",
  "pending-delta", "baseline-coverage",
  "baseline-confidence", "baseline-model-name", "baseline-model-versions",
  "current-model-id", "last-trained-at", "dataset-total", "dataset-eligible",
  "dataset-not-research", "dataset-unlicensed", "target-distribution",
  "watermark-from", "watermark-to",
  "next-version-state", "training-pipeline", "batch-ledger", "training-history-chart",
  "training-ledger", "toast-region",
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
    const payload = await request("/api/feeds");
    state.feeds = payload.feeds || [];
    const initialFeed = state.feeds.find((feed) => feed.adaptive_model) || state.feeds[0];
    if (initialFeed) state.selectedFeedIds.add(initialFeed.feed_id);
    renderFeedChecklist();
    renderPipeline([]);
    await refreshSelectedDashboards();
  } catch (error) {
    showToast(error.message);
  }
}

function renderFeedChecklist() {
  const fragment = document.createDocumentFragment();
  for (const feed of state.feeds) {
    const label = document.createElement("label");
    label.className = "feed-check";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = feed.feed_id;
    checkbox.checked = state.selectedFeedIds.has(feed.feed_id);
    checkbox.disabled = state.busy;
    const name = document.createElement("strong");
    name.textContent = feed.name;
    const detail = document.createElement("small");
    detail.textContent = feed.adaptive_model ? "deployed model" : "baseline only";
    checkbox.addEventListener("change", async () => {
      if (checkbox.checked) state.selectedFeedIds.add(feed.feed_id);
      else state.selectedFeedIds.delete(feed.feed_id);
      await refreshSelectedDashboards();
    });
    label.append(checkbox, name, detail);
    fragment.append(label);
  }
  elements["training-feeds"].replaceChildren(fragment);
}

async function refreshSelectedDashboards() {
  const selected = selectedFeeds();
  if (!selected.length) {
    state.dashboards.clear();
    renderAggregate();
    return;
  }
  elements["training-status-title"].textContent = "Loading selected datasets";
  elements["training-status-detail"].textContent = "Reading model versions, watermarks, and eligible labels.";
  const entries = await Promise.all(selected.map(async (feed) => [
    feed.feed_id,
    await request(`/api/feeds/${encodeURIComponent(feed.feed_id)}/training`),
  ]));
  state.dashboards = new Map(entries);
  renderAggregate();
}

function renderAggregate() {
  const selected = selectedFeeds();
  const dashboards = selected.map((feed) => ({ feed, dashboard: state.dashboards.get(feed.feed_id) })).filter((entry) => entry.dashboard);
  const batches = dashboards.map((entry) => entry.dashboard.live_batch || {});
  const pending = batches.reduce((sum, batch) => sum + Number(batch.eligible_samples || 0), 0);
  const baseline = aggregateEvaluation(batches.map((batch) => batch.baseline));
  const deployed = aggregateEvaluation(batches.map((batch) => batch.deployed));
  const delta = finiteDifference(deployed.accuracy, baseline.accuracy);
  const targetNames = new Set(dashboards.flatMap(({ dashboard }) => Object.keys(dashboard.dataset?.target_distribution || {})));
  const models = dashboards.map(({ dashboard }) => dashboard.model).filter(Boolean);
  const datasets = dashboards.map(({ dashboard }) => dashboard.dataset || {});

  elements["selected-feed-count"].textContent = String(selected.length);
  elements["pending-samples"].textContent = String(pending);
  elements["pending-targets"].textContent = String(targetNames.size);
  elements["pending-baseline"].textContent = formatPercent(baseline.accuracy);
  elements["baseline-counts"].textContent = formatCounts(baseline);
  elements["pending-deployed"].textContent = formatPercent(deployed.accuracy);
  elements["deployed-counts"].textContent = formatCounts(deployed);
  elements["pending-delta"].textContent = formatDelta(delta);
  elements["pending-delta"].dataset.tone = delta != null && delta >= 0 ? "positive" : "negative";
  elements["baseline-coverage"].textContent = formatPercent(baseline.coverage);
  elements["baseline-confidence"].textContent = formatPercent(baseline.mean_confidence);
  elements["baseline-model-name"].textContent = uniqueText(dashboards.map(({ dashboard }) => `${dashboard.baseline_model?.name || "SpeciesNet"} ${dashboard.baseline_model?.engine_version || ""}`));
  elements["baseline-model-versions"].textContent = uniqueText(dashboards.flatMap(({ dashboard }) => dashboard.baseline_model?.prediction_versions || []));
  elements["current-model-id"].textContent = models.length === 1 ? shortVersion(models[0].model_id) : models.length ? `${models.length} feed-specific models` : "Baseline only";
  elements["last-trained-at"].textContent = formatTime(latestTime(models.map((model) => model.trained_at)));
  elements["dataset-total"].textContent = String(sumField(datasets, "total_observations"));
  elements["dataset-eligible"].textContent = String(sumField(datasets, "eligible_labels"));
  elements["dataset-not-research"].textContent = String(sumField(datasets, "excluded_not_research"));
  elements["dataset-unlicensed"].textContent = String(sumField(datasets, "excluded_unlicensed"));
  elements["target-distribution"].textContent = targetDistributionText(dashboards);

  const windows = batches.filter((batch) => batch.window_to);
  elements["watermark-from"].textContent = windows.length === 1 ? formatTime(windows[0].window_from) : windows.length ? `${windows.length} feed-specific watermarks` : "No pending window";
  elements["watermark-to"].textContent = windows.length === 1 ? formatTime(windows[0].window_to) : windows.length ? `${windows.length} newest-label timestamps` : "Awaiting data";
  elements["next-version-state"].textContent = pending ? `${selected.length} selected · ${pending} labels ready` : "Awaiting unseen labels";
  elements["run-training"].disabled = state.busy || pending === 0;
  elements["run-training"].textContent = pending ? `Train ${selected.length} selected feed${selected.length === 1 ? "" : "s"} on ${pending} labels` : selected.length ? "Awaiting new labeled data" : "Select a dataset";
  elements["training-status-title"].textContent = selected.length ? `${selected.length} dataset${selected.length === 1 ? "" : "s"} selected` : "Select protected-area datasets";
  elements["training-status-detail"].textContent = pending ? "Every image in the manifest below will be consumed; models remain feed-specific." : "No selected feed currently has an eligible pending batch.";

  renderBatch(dashboards);
  renderLedger(dashboards);
  drawHistory(dashboards);
}

async function startTraining() {
  const queue = selectedFeeds().filter((feed) => Number(state.dashboards.get(feed.feed_id)?.live_batch?.eligible_samples || 0) > 0);
  if (!queue.length) return;
  setBusy(true);
  const completed = [];
  try {
    for (const feed of queue) {
      elements["training-status-title"].textContent = `Training ${feed.name}`;
      const payload = await request(`/api/feeds/${encodeURIComponent(feed.feed_id)}/train`, { method: "POST", body: "{}" });
      const job = await waitForJob(payload.job.job_id, feed.name);
      if (job.state !== "completed") throw new Error(job.error || `${feed.name} training failed`);
      completed.push(job);
    }
    await refreshSelectedDashboards();
    elements["training-progress"].value = 100;
  } catch (error) {
    elements["training-status-title"].textContent = "Training stopped";
    elements["training-status-detail"].textContent = error.message;
    showToast(error.message);
  } finally {
    setBusy(false);
  }
  if (completed.length) {
    elements["training-progress"].value = 100;
    elements["training-status-title"].textContent = "Selected training jobs complete";
    elements["training-status-detail"].textContent = completed.map((job) => `${feedName(job.feed_id)} → ${shortVersion(job.details.trained_model_id)}`).join(" · ");
  }
}

async function waitForJob(jobId, feedNameValue) {
  while (true) {
    const payload = await request(`/api/jobs/${encodeURIComponent(jobId)}`);
    const job = payload.job;
    renderPipeline(job.details?.pipeline || [], feedNameValue);
    const active = (job.details?.pipeline || []).find((stage) => stage.state === "running");
    elements["training-status-detail"].textContent = active ? `${active.label} · ${active.detail || "running"}` : `${job.processed} / ${job.total || "?"} observations · ${job.state}`;
    elements["training-progress"].value = pipelinePercent(job.details?.pipeline || []);
    if (!["pending", "running"].includes(job.state)) return job;
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
}

function renderPipeline(stages, feedNameValue = null) {
  const byId = new Map(stages.map((stage) => [stage.id, stage]));
  const fragment = document.createDocumentFragment();
  for (const [id, label] of PIPELINE_STAGES) {
    const stage = byId.get(id) || { id, label, state: "pending", processed: 0, total: null, detail: null };
    const item = document.createElement("li");
    item.dataset.state = stage.state;
    const title = document.createElement("strong");
    title.textContent = stage.label || label;
    const detail = document.createElement("span");
    detail.textContent = stage.detail || (feedNameValue ? `${feedNameValue} · pending` : "Pending");
    const count = document.createElement("small");
    count.textContent = stage.total == null ? stage.state : `${stage.processed || 0} / ${stage.total}`;
    item.append(title, detail, count);
    fragment.append(item);
  }
  elements["training-pipeline"].replaceChildren(fragment);
}

function renderBatch(entries) {
  const fragment = document.createDocumentFragment();
  for (const { feed, dashboard } of entries) {
    for (const sample of dashboard.live_batch?.samples || []) {
      const row = document.createElement("tr");
      appendCell(row, feed.name);
      const imageCell = document.createElement("td");
      const link = document.createElement("a");
      link.href = `/api/images/${sample.photo_id}`;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = `Photo ${sample.photo_id}`;
      imageCell.append(link);
      row.append(imageCell);
      const deployedMetric = sample.deployed_margin == null
        ? formatPercent(sample.deployed_confidence)
        : `Margin ${Number(sample.deployed_margin).toFixed(3)}`;
      for (const value of [formatTime(sample.created_at), sample.obtained_common_name || sample.obtained_scientific_name, `${displayLabel(sample.baseline_label)} · ${formatPercent(sample.baseline_confidence)}`, sample.deployed_label ? `${displayLabel(sample.deployed_label)} · ${deployedMetric}` : "No deployed model"]) appendCell(row, value);
      fragment.append(row);
    }
  }
  if (!fragment.childNodes.length) appendEmptyRow(fragment, 6, "No unseen eligible labels are waiting for training.");
  elements["batch-ledger"].replaceChildren(fragment);
}

function renderLedger(entries) {
  const fragment = document.createDocumentFragment();
  for (const { feed, dashboard } of entries) {
    const runs = (dashboard.runs || []).filter((run) => run.details?.evaluated_model_id && run.details?.trained_model_id);
    for (const run of runs) {
      const details = run.details;
      const row = document.createElement("tr");
      for (const value of [feed.name, formatTime(run.finished_at), shortVersion(details.evaluated_model_id), details.training_samples ?? 0, formatPercent(details.baseline_accuracy), formatPercent(details.deployed_accuracy), shortVersion(details.trained_model_id), formatPercent(details.training_agreement), formatDuration(details.duration_seconds), formatTime(details.watermark || details.watermark_to)]) appendCell(row, value);
      fragment.append(row);
    }
  }
  if (!fragment.childNodes.length) appendEmptyRow(fragment, 10, "No test-then-train runs yet.");
  elements["training-ledger"].replaceChildren(fragment);
}

function drawHistory(entries) {
  const runs = entries.flatMap(({ dashboard }) => dashboard.runs || []).filter((run) => run.details?.evaluated_model_id).reverse();
  const canvas = elements["training-history-chart"];
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfcf8";
  context.fillRect(0, 0, width, height);
  const margin = { left: 58, right: 24, top: 30, bottom: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  context.font = "12px Aptos";
  for (let step = 0; step <= 4; step += 1) {
    const y = margin.top + plotHeight - step / 4 * plotHeight;
    context.strokeStyle = "#d2d9d2";
    context.beginPath(); context.moveTo(margin.left, y); context.lineTo(width - margin.right, y); context.stroke();
    context.fillStyle = "#5f6b63"; context.fillText(`${step * 25}%`, 12, y + 4);
  }
  drawSeries(context, runs.map((run) => run.details), "baseline_accuracy", "#185b3b", margin, plotWidth, plotHeight);
  drawSeries(context, runs.map((run) => run.details), "deployed_accuracy", "#d49a2a", margin, plotWidth, plotHeight);
  context.fillStyle = "#185b3b"; context.fillRect(width - 230, 13, 12, 12);
  context.fillStyle = "#16201a"; context.fillText("Baseline", width - 212, 23);
  context.fillStyle = "#d49a2a"; context.fillRect(width - 125, 13, 12, 12);
  context.fillStyle = "#16201a"; context.fillText("Deployed", width - 107, 23);
}

function drawSeries(context, points, field, color, margin, plotWidth, plotHeight) {
  const values = points.filter((point) => point[field] != null);
  if (!values.length) return;
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 3;
  context.beginPath();
  values.forEach((point, index) => {
    const x = margin.left + (values.length === 1 ? plotWidth / 2 : index / (values.length - 1) * plotWidth);
    const y = margin.top + plotHeight - Math.max(0, Math.min(1, Number(point[field]))) * plotHeight;
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
}

function aggregateEvaluation(values) {
  const available = values.filter((value) => value?.accuracy != null);
  const samples = available.reduce((sum, value) => sum + Number(value.samples || 0), 0);
  const correct = available.reduce((sum, value) => sum + Number(value.correct || 0), 0);
  const errors = available.reduce((sum, value) => sum + Number(value.errors || 0), 0);
  return {
    samples,
    correct,
    errors,
    accuracy: samples ? correct / samples : null,
    coverage: weightedMetric(available, "coverage"),
    mean_confidence: weightedMetric(available, "mean_confidence"),
  };
}

function weightedMetric(values, field) {
  const usable = values.filter((value) => value[field] != null && Number(value.samples) > 0);
  const samples = usable.reduce((sum, value) => sum + Number(value.samples), 0);
  return samples ? usable.reduce((sum, value) => sum + Number(value[field]) * Number(value.samples), 0) / samples : null;
}

function selectedFeeds() { return state.feeds.filter((feed) => state.selectedFeedIds.has(feed.feed_id)); }
function feedName(feedId) { return state.feeds.find((feed) => feed.feed_id === feedId)?.name || feedId; }
function sumField(rows, field) { return rows.reduce((sum, row) => sum + Number(row[field] || 0), 0); }
function uniqueText(values) { const unique = [...new Set(values.filter(Boolean))]; return unique.length ? unique.join(", ") : "--"; }
function formatCounts(value) { return !value?.samples || value.correct == null ? "--" : `${value.correct} / ${value.errors}`; }
function targetDistributionText(entries) { const counts = new Map(); for (const { dashboard } of entries) for (const [target, count] of Object.entries(dashboard.dataset?.target_distribution || {})) counts.set(target, (counts.get(target) || 0) + Number(count)); return [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])).map(([target, count]) => `${target} (${count})`).join(", ") || "--"; }
function latestTime(values) { const dates = values.filter(Boolean).map((value) => new Date(value)).filter((value) => !Number.isNaN(value.getTime())); return dates.length ? Math.max(...dates.map(Number)) : null; }
function appendCell(row, value) { const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell); }
function appendEmptyRow(fragment, colSpan, message) { const row = document.createElement("tr"); const cell = document.createElement("td"); cell.colSpan = colSpan; cell.textContent = message; row.append(cell); fragment.append(row); }
function pipelinePercent(stages) { return stages.length ? Math.round(stages.reduce((sum, stage) => sum + (stage.state === "completed" ? 1 : stage.state === "running" ? .5 : 0), 0) / stages.length * 100) : 0; }
function finiteDifference(left, right) { return left != null && right != null ? Number(left) - Number(right) : null; }
function displayLabel(value) { const parts = String(value || "Unavailable").split(";").map((part) => part.trim()).filter(Boolean); return parts.at(-1) || "Unavailable"; }
function shortVersion(value) { if (!value) return "Not available"; const text = String(value); return text.length > 30 ? `${text.slice(0, 18)}…${text.slice(-8)}` : text; }
function formatTime(value) { if (!value) return "--"; const date = value instanceof Date || typeof value === "number" ? new Date(value) : new Date(String(value)); return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString(); }
function formatPercent(value) { return value == null || value === "" ? "--" : `${(Number(value) * 100).toFixed(1)}%`; }
function formatDelta(value) { if (value == null) return "--"; const points = Number(value) * 100; return `${points >= 0 ? "+" : ""}${points.toFixed(1)} pts`; }
function formatDuration(value) { const seconds = Number(value); return Number.isFinite(seconds) ? seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s` : "--"; }

function setBusy(busy) {
  state.busy = busy;
  for (const checkbox of elements["training-feeds"].querySelectorAll("input")) {
    checkbox.disabled = busy;
  }
  if (!busy) renderAggregate();
  else elements["run-training"].disabled = true;
}

elements["run-training"].addEventListener("click", startTraining);
await initialize();

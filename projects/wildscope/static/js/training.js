const state = { feeds: [], feedId: null, jobId: null, timer: null };

const elements = Object.fromEntries([
  "training-feed", "run-training", "training-status-title", "training-status-detail",
  "training-progress", "current-model-id", "pending-samples", "pending-baseline",
  "pending-deployed", "pending-delta", "watermark-from", "watermark-to",
  "next-version-state", "batch-ledger", "training-history-chart", "training-ledger",
  "toast-region",
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
    for (const feed of state.feeds) {
      const option = document.createElement("option");
      option.value = feed.feed_id;
      option.textContent = `${feed.name} · ${feed.country}`;
      elements["training-feed"].append(option);
    }
    const initialFeed = state.feeds.find((feed) => feed.adaptive_model) || state.feeds[0];
    if (initialFeed) {
      elements["training-feed"].value = initialFeed.feed_id;
      await selectFeed();
    }
  } catch (error) {
    showToast(error.message);
  }
}

async function selectFeed() {
  state.feedId = elements["training-feed"].value || null;
  elements["run-training"].disabled = true;
  if (!state.feedId) return;
  elements["training-status-title"].textContent = state.feeds.find((feed) => feed.feed_id === state.feedId)?.name || state.feedId;
  elements["training-status-detail"].textContent = "Loading data received since the deployed model watermark.";
  await loadDashboard();
}

async function startTraining() {
  if (!state.feedId) return;
  setBusy(true);
  elements["training-status-title"].textContent = "Starting incremental training";
  elements["training-status-detail"].textContent = "Requesting observation batches after the last watermark.";
  try {
    const payload = await request(`/api/feeds/${encodeURIComponent(state.feedId)}/train`, {
      method: "POST", body: "{}",
    });
    state.jobId = payload.job.job_id;
    pollJob();
  } catch (error) {
    setBusy(false);
    elements["training-status-title"].textContent = "Training did not start";
    elements["training-status-detail"].textContent = error.message;
    showToast(error.message);
  }
}

async function pollJob() {
  try {
    const payload = await request(`/api/jobs/${encodeURIComponent(state.jobId)}`);
    const job = payload.job;
    const percent = job.total ? Math.round(job.processed / job.total * 100) : 5;
    elements["training-progress"].value = Math.max(0, Math.min(100, percent));
    elements["training-status-title"].textContent = "Testing deployed model, then training next version";
    elements["training-status-detail"].textContent = `${job.processed} / ${job.total || "?"} observations · ${job.state}`;
    if (["pending", "running"].includes(job.state)) {
      state.timer = setTimeout(pollJob, 1200);
      return;
    }
    setBusy(false);
    if (job.state === "failed") {
      elements["training-progress"].value = 0;
      elements["training-status-title"].textContent = "Training stopped";
      elements["training-status-detail"].textContent = job.error || "Unknown worker error";
      showToast(job.error || "Training failed");
      return;
    }
    elements["training-progress"].value = 100;
    await loadDashboard();
    elements["training-status-title"].textContent = "Training complete";
    elements["training-status-detail"].textContent = `${shortVersion(job.details.evaluated_model_id)} scored ${formatPercent(job.details.deployed_accuracy)} on ${job.details.training_samples} new labels · ${shortVersion(job.details.trained_model_id)} trained and awaits the next batch.`;
  } catch (error) {
    state.timer = setTimeout(pollJob, 1600);
  }
}

async function loadDashboard() {
  try {
    const dashboard = await request(`/api/feeds/${encodeURIComponent(state.feedId)}/training`);
    renderDashboard(dashboard);
  } catch (error) {
    showToast(error.message);
  }
}

function renderDashboard(dashboard) {
  const batch = dashboard.live_batch || {};
  const baseline = batch.baseline || {};
  const deployed = batch.deployed || {};
  const delta = finiteDifference(deployed.accuracy, baseline.accuracy);
  elements["current-model-id"].textContent = shortVersion(batch.evaluated_model_id);
  elements["pending-samples"].textContent = String(batch.eligible_samples || 0);
  elements["pending-baseline"].textContent = formatPercent(baseline.accuracy);
  elements["pending-deployed"].textContent = formatPercent(deployed.accuracy);
  elements["pending-delta"].textContent = formatDelta(delta);
  elements["pending-delta"].dataset.tone = Number(delta) >= 0 ? "positive" : "negative";
  elements["watermark-from"].textContent = batch.window_from ? formatTime(batch.window_from) : "First model";
  elements["watermark-to"].textContent = batch.window_to ? formatTime(batch.window_to) : "Awaiting data";
  elements["next-version-state"].textContent = batch.eligible_samples
    ? batch.bootstrap_migration
      ? `Ready to bootstrap on ${batch.eligible_samples} labels`
      : `Ready to train on ${batch.eligible_samples} labels`
    : "Awaiting unseen labels";
  elements["run-training"].disabled = !batch.eligible_samples;
  elements["run-training"].textContent = batch.eligible_samples
    ? batch.bootstrap_migration
      ? `Bootstrap lifecycle on ${batch.eligible_samples} labels`
      : `Train next version on ${batch.eligible_samples} labels`
    : "Awaiting new labeled data";
  elements["training-status-detail"].textContent = batch.eligible_samples
    ? batch.bootstrap_migration
      ? "Legacy model detected. Existing research-grade labels will initialize test-then-train v1; SpeciesNet is the auditable baseline for this one migration."
      : "The deployed model has already been scored on this batch. Training will consume every listed image."
    : "No eligible research-grade labels have arrived since the current model watermark.";
  renderBatch(batch.samples || []);
  renderLedger(dashboard.runs || []);
  drawHistory(dashboard.runs || []);
}

function renderBatch(samples) {
  const fragment = document.createDocumentFragment();
  for (const sample of samples) {
    const row = document.createElement("tr");
    const imageCell = document.createElement("td");
    const link = document.createElement("a");
    link.href = `/api/images/${sample.photo_id}`;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = `Photo ${sample.photo_id}`;
    imageCell.append(link);
    row.append(imageCell);
    for (const value of [
      formatTime(sample.created_at),
      sample.obtained_common_name || sample.obtained_scientific_name,
      `${displayLabel(sample.baseline_label)} · ${formatPercent(sample.baseline_confidence)}`,
      sample.deployed_label
        ? `${displayLabel(sample.deployed_label)} · ${formatPercent(sample.deployed_confidence)}`
        : "No deployed model",
    ]) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    }
    fragment.append(row);
  }
  if (!samples.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "No unseen eligible labels are waiting for training.";
    row.append(cell);
    fragment.append(row);
  }
  elements["batch-ledger"].replaceChildren(fragment);
}

function renderLedger(runs) {
  const fragment = document.createDocumentFragment();
  const versionedRuns = runs.filter(
    (run) => run.details?.evaluated_model_id && run.details?.trained_model_id,
  );
  for (const run of versionedRuns) {
    const details = run.details || {};
    const row = document.createElement("tr");
    const values = [
      formatTime(run.finished_at), shortVersion(details.evaluated_model_id),
      details.training_samples ?? 0, formatPercent(details.baseline_accuracy),
      formatPercent(details.deployed_accuracy), shortVersion(details.trained_model_id),
      formatPercent(details.training_agreement), formatDuration(details.duration_seconds),
      formatTime(details.watermark || details.watermark_to),
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    }
    fragment.append(row);
  }
  if (!versionedRuns.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 9;
    cell.textContent = "No test-then-train runs yet.";
    row.append(cell);
    fragment.append(row);
  }
  elements["training-ledger"].replaceChildren(fragment);
}

function drawHistory(runs) {
  const canvas = elements["training-history-chart"];
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfcf8";
  context.fillRect(0, 0, width, height);
  const ordered = [...runs].filter((run) => run.details?.evaluated_model_id).reverse();
  const points = ordered.map((run) => run.details || {});
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
  drawSeries(context, points, "baseline_accuracy", "#185b3b", margin, plotWidth, plotHeight);
  drawSeries(context, points, "deployed_accuracy", "#d49a2a", margin, plotWidth, plotHeight);
  context.fillStyle = "#185b3b"; context.fillRect(width - 260, 13, 12, 12);
  context.fillStyle = "#16201a"; context.fillText("Baseline", width - 242, 23);
  context.fillStyle = "#d49a2a"; context.fillRect(width - 150, 13, 12, 12);
  context.fillStyle = "#16201a"; context.fillText("Deployed", width - 132, 23);
}

function drawSeries(context, points, field, color, margin, plotWidth, plotHeight) {
  if (!points.length) return;
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 3;
  context.beginPath();
  points.forEach((point, index) => {
    const x = margin.left + (points.length === 1 ? plotWidth / 2 : index / (points.length - 1) * plotWidth);
    const value = Math.max(0, Math.min(1, Number(point[field]) || 0));
    const y = margin.top + plotHeight - value * plotHeight;
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  points.forEach((point, index) => {
    const x = margin.left + (points.length === 1 ? plotWidth / 2 : index / (points.length - 1) * plotWidth);
    const value = Math.max(0, Math.min(1, Number(point[field]) || 0));
    const y = margin.top + plotHeight - value * plotHeight;
    context.beginPath(); context.arc(x, y, 5, 0, Math.PI * 2); context.fill();
  });
}

function setBusy(busy) {
  elements["run-training"].disabled = busy || !state.feedId;
  elements["training-feed"].disabled = busy;
}

function finiteDifference(left, right) {
  if (left == null || right == null || left === "" || right === "") return null;
  const leftNumber = Number(left);
  const rightNumber = Number(right);
  return Number.isFinite(leftNumber) && Number.isFinite(rightNumber)
    ? leftNumber - rightNumber
    : null;
}

function displayLabel(value) {
  const parts = String(value || "Unavailable").split(";").map((part) => part.trim()).filter(Boolean);
  return parts.at(-1) || "Unavailable";
}

function shortVersion(value) {
  if (!value) return "Not available";
  const text = String(value);
  return text.length > 30 ? `${text.slice(0, 18)}…${text.slice(-8)}` : text;
}

function formatTime(value) {
  if (!value) return "--";
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
  return seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

elements["training-feed"].addEventListener("change", selectFeed);
elements["run-training"].addEventListener("click", startTraining);
await initialize();

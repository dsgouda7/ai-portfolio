export const PIPELINE_STAGES = [
  ["source_acquisition", "Source acquisition"],
  ["frame_validation", "Frame validation"],
  ["vehicle_detection", "Vehicle detection"],
  ["track_association", "Track association"],
  ["crop_quality", "Crop quality"],
  ["frame_classification", "Frame classification"],
  ["track_fusion", "Track fusion"],
  ["calibration", "Calibration"],
  ["hierarchy_decision", "Hierarchy decision"],
  ["privacy_render", "Privacy render"],
];

const VALID_STATES = new Set(["pending", "running", "completed", "skipped", "warning", "failed"]);

function durationLabel(duration) {
  const value = Number(duration);
  if (!Number.isFinite(value)) return "--";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(value < 10 ? 1 : 0)} ms`;
}

export class PipelineGraph {
  constructor(container, onSelect) {
    this.container = container;
    this.onSelect = onSelect;
    this.nodes = new Map();
    this.events = new Map();
    this.selectedStage = PIPELINE_STAGES[0][0];
    this.#build();
  }

  #build() {
    this.container.replaceChildren();
    PIPELINE_STAGES.forEach(([stage, label], index) => {
      const wrapper = document.createElement("div");
      wrapper.className = "pipeline-stage-wrap";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "pipeline-stage";
      button.dataset.stage = stage;
      button.dataset.status = "pending";
      button.dataset.testid = `pipeline-stage-${stage}`;
      button.setAttribute("aria-label", `${label}: pending`);
      button.setAttribute("aria-pressed", String(stage === this.selectedStage));

      const number = document.createElement("span");
      number.className = "pipeline-stage__index";
      number.textContent = String(index + 1).padStart(2, "0");
      const name = document.createElement("span");
      name.className = "pipeline-stage__label";
      name.textContent = label;
      const meta = document.createElement("span");
      meta.className = "pipeline-stage__meta";
      const state = document.createElement("span");
      state.textContent = "Pending";
      const dot = document.createElement("span");
      dot.className = "pipeline-stage__status";
      dot.setAttribute("aria-hidden", "true");
      meta.append(state, dot);
      button.append(number, name, meta);
      button.addEventListener("click", () => this.select(stage));
      wrapper.append(button);
      this.container.append(wrapper);
      this.nodes.set(stage, { button, state, label });
    });
    this.select(this.selectedStage);
  }

  reset() {
    this.events.clear();
    for (const { button, state, label } of this.nodes.values()) {
      button.dataset.status = "pending";
      button.setAttribute("aria-label", `${label}: pending`);
      state.textContent = "Pending";
    }
    this.select(PIPELINE_STAGES[0][0]);
  }

  update(event) {
    const node = this.nodes.get(event.stage);
    if (!node) return;
    const status = VALID_STATES.has(event.status) ? event.status : "warning";
    const normalized = { ...event, status };
    this.events.set(event.stage, normalized);
    node.button.dataset.status = status;
    node.button.setAttribute("aria-label", `${node.label}: ${status}, ${durationLabel(event.duration_ms)}`);
    node.state.textContent = event.duration_ms == null ? status : durationLabel(event.duration_ms);
    if (event.stage === this.selectedStage || status === "failed" || status === "warning") {
      this.select(event.stage);
    }
  }

  select(stage) {
    if (!this.nodes.has(stage)) return;
    this.selectedStage = stage;
    for (const [name, node] of this.nodes.entries()) {
      node.button.setAttribute("aria-pressed", String(name === stage));
    }
    const node = this.nodes.get(stage);
    this.onSelect?.({
      stage,
      label: node.label,
      event: this.events.get(stage) || { stage, status: "pending", duration_ms: null },
    });
  }
}

export function formatDuration(duration) {
  return durationLabel(duration);
}

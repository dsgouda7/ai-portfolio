"""Calibrate SmolLM2-135M objective costs on the local CPU."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from importlib.metadata import version
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
BASE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
INSTRUCT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
INSTRUCT_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SEED = 2026
WARM_STEPS = 3
MEASURED_STEPS = 5
ADAPTIVE_MILESTONES = (10, 25, 50, 100)
CPU_BUDGET_MINUTES = 360
GPU_BUDGET_MINUTES = 20
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
SYSTEM_PROMPT = "You are Riverside House's concise fiction-writing assistant."
SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = SCRIPT_DIR / "content" / "the-weight-of-distant-light"
DEFAULT_OUTPUT = SCRIPT_DIR / "smollm2-135m-calibration.json"


def adaptive_milestones(
	seconds_per_step: float,
	*,
	device: str = "cpu",
	full_pass_steps: int | None = None,
) -> list[int]:
	"""Return ordered stops whose measured training time fits, excluding evaluation."""
	budget_minutes = GPU_BUDGET_MINUTES if device == "cuda" else CPU_BUDGET_MINUTES
	candidates = list(ADAPTIVE_MILESTONES)
	if full_pass_steps is not None:
		candidates.append(full_pass_steps)
	return [
		step
		for step in sorted(set(candidates))
		if seconds_per_step * step / 60 <= budget_minutes
	]


class RssSampler:
	def __init__(self) -> None:
		self.process = psutil.Process()
		self.baseline = self.process.memory_info().rss
		self.peak = self.baseline
		self.stop_event = threading.Event()
		self.thread = threading.Thread(target=self._sample, daemon=True)

	def _sample(self) -> None:
		while not self.stop_event.wait(0.02):
			self.peak = max(self.peak, self.process.memory_info().rss)

	def __enter__(self) -> "RssSampler":
		self.thread.start()
		return self

	def __exit__(self, *_args: object) -> None:
		self.stop_event.set()
		self.thread.join()
		self.peak = max(self.peak, self.process.memory_info().rss)

	def report(self) -> dict[str, float]:
		divisor = 1024**2
		return {
			"baseline_rss_mb": round(self.baseline / divisor, 2),
			"peak_rss_mb": round(self.peak / divisor, 2),
			"peak_delta_mb": round((self.peak - self.baseline) / divisor, 2),
		}


def hardware_report() -> dict[str, object]:
	memory = psutil.virtual_memory()
	return {
		"platform": platform.platform(),
		"processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
		"physical_cores": psutil.cpu_count(logical=False),
		"logical_cores": psutil.cpu_count(logical=True),
		"total_ram_gb": round(memory.total / 1024**3, 2),
		"available_ram_gb_at_start": round(memory.available / 1024**3, 2),
		"torch_threads": torch.get_num_threads(),
		"packages": {
			package: version(package)
			for package in ("torch", "transformers", "peft", "trl", "psutil")
		},
	}


def load_paragraphs(path: Path, min_length: int = 120) -> list[str]:
	return [
		paragraph.strip().replace("\n", " ")
		for paragraph in path.read_text(encoding="utf-8").split("\n\n")
		if len(paragraph.strip()) >= min_length
	]


def first_sentence(text: str, limit: int = 260) -> str:
	match = re.search(r"^.*?[.!?](?=\s|$)", text.strip())
	sentence = match.group(0) if match else text.strip()
	return sentence[:limit].strip()


def evenly_spaced(items: list[dict[str, str]], count: int) -> list[dict[str, str]]:
	if len(items) < count:
		raise ValueError(f"Need {count} examples, found {len(items)}")
	if count == 1:
		return [items[0]]
	indexes = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
	return [items[index] for index in indexes]


def build_examples() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
	chapter_files = sorted(CORPUS_DIR.glob("chapter-*.txt"))
	if len(chapter_files) != 40:
		raise FileNotFoundError(f"Expected 40 Aria chapters under {CORPUS_DIR}, found {len(chapter_files)}")

	def chapter_examples(files: list[Path]) -> list[dict[str, str]]:
		examples = []
		for path in files:
			paragraphs = load_paragraphs(path)
			if len(paragraphs) < 2:
				continue
			context = paragraphs[0][:360].strip()
			response = first_sentence(paragraphs[1])
			examples.append(
				{
					"chapter": path.name,
					"text": paragraphs[0],
					"instruction": f"Continue this Aria passage in one sentence and stop: {context}",
					"chosen": response,
					"rejected": "The moment remained unchanged, and everyone stayed where they were.",
				}
			)
		return examples

	train = evenly_spaced(chapter_examples(chapter_files[:-4]), WARM_STEPS + MEASURED_STEPS)
	holdout = chapter_examples(chapter_files[-4:])
	return train, holdout


def load_tokenizer(model_id: str, revision: str):
	tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	return tokenizer


def tensor_batch(input_ids: list[int], labels: list[int], max_length: int, pad_id: int):
	if len(input_ids) > max_length:
		input_ids = input_ids[:max_length]
		labels = labels[:max_length]
	padding = max_length - len(input_ids)
	return {
		"input_ids": torch.tensor([input_ids + [pad_id] * padding]),
		"attention_mask": torch.tensor([[1] * len(input_ids) + [0] * padding]),
		"labels": torch.tensor([labels + [-100] * padding]),
	}


def encode_cpt(tokenizer, text: str, max_length: int = 64):
	ids = tokenizer(text + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
	ids = ids[:max_length]
	return tensor_batch(ids, ids.copy(), max_length, tokenizer.pad_token_id)


def bounded_prompt_ids(tokenizer, instruction: str, budget: int) -> list[int]:
	context = instruction
	while True:
		messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": context},
		]
		prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
		if len(prompt_ids) <= budget:
			return prompt_ids
		context = context[:-24].rstrip()
		if not context:
			raise ValueError("Could not fit the instruction within the prompt budget")


def encode_completion(
	tokenizer,
	instruction: str,
	response: str,
	max_length: int,
	prompt_budget: int,
):
	prompt_ids = bounded_prompt_ids(tokenizer, instruction, prompt_budget)
	response_budget = max_length - len(prompt_ids)
	response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
	response_ids = response_ids[: max(1, response_budget - 1)] + [tokenizer.eos_token_id]
	input_ids = prompt_ids + response_ids
	labels = [-100] * len(prompt_ids) + response_ids
	return tensor_batch(input_ids, labels, max_length, tokenizer.pad_token_id)


def lora_config() -> LoraConfig:
	return LoraConfig(
		r=8,
		lora_alpha=16,
		lora_dropout=0.05,
		bias="none",
		task_type=TaskType.CAUSAL_LM,
		target_modules=LORA_TARGET_MODULES,
	)


def parameter_report(model) -> dict[str, int]:
	return {
		"total": sum(parameter.numel() for parameter in model.parameters()),
		"trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
	}


def timed_optimizer_steps(model, optimizer, batches, step_function) -> tuple[list[float], list[float]]:
	warm_times = []
	measured_times = []
	for index, batch in enumerate(batches):
		started = time.perf_counter()
		loss = step_function(model, optimizer, batch)
		elapsed = time.perf_counter() - started
		destination = warm_times if index < WARM_STEPS else measured_times
		destination.append(elapsed)
		print(
			f"step {index + 1}/{len(batches)} | {elapsed:.3f}s | loss={loss:.5f} | "
			f"{'warm-up' if index < WARM_STEPS else 'measured'}",
			flush=True,
		)
	return warm_times, measured_times


def causal_step(model, optimizer, batch) -> float:
	model.train()
	optimizer.zero_grad(set_to_none=True)
	loss = model(**batch).loss
	loss.backward()
	optimizer.step()
	return float(loss.detach())


def completion_logps(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
	logits = model(
		input_ids=batch["input_ids"],
		attention_mask=batch["attention_mask"],
	).logits[:, :-1, :]
	labels = batch["labels"][:, 1:]
	mask = labels.ne(-100)
	token_logps = F.log_softmax(logits, dim=-1).gather(
		dim=-1,
		index=labels.masked_fill(~mask, 0).unsqueeze(-1),
	).squeeze(-1)
	return (token_logps * mask).sum(dim=-1)


def stack_pair(chosen: dict[str, torch.Tensor], rejected: dict[str, torch.Tensor]):
	return {
		key: torch.cat([chosen[key], rejected[key]], dim=0)
		for key in ("input_ids", "attention_mask", "labels")
	}


def dpo_step(models, optimizer, batch, beta: float = 0.1) -> float:
	policy, reference = models
	policy.train()
	optimizer.zero_grad(set_to_none=True)
	policy_logps = completion_logps(policy, batch)
	with torch.no_grad():
		reference_logps = completion_logps(reference, batch)
	preference_logit = (policy_logps[0] - policy_logps[1]) - (
		reference_logps[0] - reference_logps[1]
	)
	loss = -F.logsigmoid(beta * preference_logit)
	loss.backward()
	optimizer.step()
	return float(loss.detach())


def evaluate_causal(model, batches: list[dict[str, torch.Tensor]]) -> tuple[float, float]:
	model.eval()
	started = time.perf_counter()
	with torch.no_grad():
		losses = [float(model(**batch).loss) for batch in batches]
	return statistics.mean(losses), time.perf_counter() - started


def evaluate_dpo(models, batches: list[dict[str, torch.Tensor]]) -> tuple[float, float]:
	policy, reference = models
	policy.eval()
	reference.eval()
	started = time.perf_counter()
	edges = []
	with torch.no_grad():
		for batch in batches:
			policy_logps = completion_logps(policy, batch)
			reference_logps = completion_logps(reference, batch)
			edges.append(
				float(
					(policy_logps[0] - policy_logps[1])
					- (reference_logps[0] - reference_logps[1])
				)
			)
	return statistics.mean(edges), time.perf_counter() - started


def directory_size_mb(path: Path) -> float:
	return round(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024**2, 2)


def timing_report(warm_times: list[float], measured_times: list[float]) -> dict[str, object]:
	mean_seconds = statistics.mean(measured_times)
	return {
		"warm_step_seconds": [round(value, 4) for value in warm_times],
		"measured_step_seconds": [round(value, 4) for value in measured_times],
		"mean_seconds_per_step": round(mean_seconds, 4),
		"median_seconds_per_step": round(statistics.median(measured_times), 4),
		"min_seconds_per_step": round(min(measured_times), 4),
		"max_seconds_per_step": round(max(measured_times), 4),
		"estimated_training_minutes": {
			str(milestone): round(mean_seconds * milestone / 60, 2)
			for milestone in (10, 25, 50, 100)
		},
	}


def run_stage(stage: str) -> dict[str, object]:
	set_seed(SEED)
	torch.set_num_threads(min(8, psutil.cpu_count(logical=False) or 1))
	train_examples, holdout_examples = build_examples()
	model_id = BASE_MODEL if stage == "cpt" else INSTRUCT_MODEL
	revision = BASE_REVISION if stage == "cpt" else INSTRUCT_REVISION

	with RssSampler() as memory:
		load_started = time.perf_counter()
		tokenizer = load_tokenizer(model_id, revision)
		if stage == "cpt":
			model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
			model.config.use_cache = False
			models = model
			batches = [encode_cpt(tokenizer, item["text"]) for item in train_examples]
			evaluation_batches = [encode_cpt(tokenizer, item["text"]) for item in holdout_examples]
			optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
			step_function = causal_step
			parameters = parameter_report(model)
		elif stage == "sft":
			base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
			base.config.use_cache = False
			model = get_peft_model(base, lora_config())
			models = model
			batches = [
				encode_completion(tokenizer, item["instruction"], item["chosen"], 96, 64)
				for item in train_examples
			]
			evaluation_batches = [
				encode_completion(tokenizer, item["instruction"], item["chosen"], 96, 64)
				for item in holdout_examples
			]
			optimizer = torch.optim.AdamW(
				(parameter for parameter in model.parameters() if parameter.requires_grad),
				lr=2e-4,
			)
			step_function = causal_step
			parameters = parameter_report(model)
		else:
			policy_base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
			reference_base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
			policy_base.config.use_cache = False
			reference_base.config.use_cache = False
			policy = get_peft_model(policy_base, lora_config())
			reference = get_peft_model(reference_base, lora_config())
			reference.eval()
			for parameter in reference.parameters():
				parameter.requires_grad_(False)
			models = (policy, reference)

			def encode_dpo(item: dict[str, str]):
				chosen = encode_completion(tokenizer, item["instruction"], item["chosen"], 128, 64)
				rejected = encode_completion(tokenizer, item["instruction"], item["rejected"], 128, 64)
				return stack_pair(chosen, rejected)

			batches = [encode_dpo(item) for item in train_examples]
			evaluation_batches = [encode_dpo(item) for item in holdout_examples]
			optimizer = torch.optim.AdamW(
				(parameter for parameter in policy.parameters() if parameter.requires_grad),
				lr=5e-5,
			)
			step_function = dpo_step
			parameters = parameter_report(policy)
		load_seconds = time.perf_counter() - load_started

		warm_times, measured_times = timed_optimizer_steps(
			models,
			optimizer,
			batches,
			step_function,
		)
		if stage == "dpo":
			evaluation_value, evaluation_seconds = evaluate_dpo(models, evaluation_batches)
			checkpoint_model = models[0]
			evaluation_name = "mean_heldout_preference_edge"
		else:
			evaluation_value, evaluation_seconds = evaluate_causal(models, evaluation_batches)
			checkpoint_model = models
			evaluation_name = "mean_heldout_loss"

		with tempfile.TemporaryDirectory(prefix=f"smollm2-135m-{stage}-") as temporary:
			checkpoint_path = Path(temporary)
			save_started = time.perf_counter()
			checkpoint_model.save_pretrained(checkpoint_path, safe_serialization=True)
			tokenizer.save_pretrained(checkpoint_path)
			checkpoint_save_seconds = time.perf_counter() - save_started
			checkpoint_size = directory_size_mb(checkpoint_path)

	result = {
		"stage": stage,
		"model": {"id": model_id, "revision": revision},
		"sequence_length": {"cpt": 64, "sft": 96, "dpo": 128}[stage],
		"warm_steps": WARM_STEPS,
		"measured_steps": MEASURED_STEPS,
		"train_chapters_sampled": [item["chapter"] for item in train_examples],
		"holdout_chapters": [item["chapter"] for item in holdout_examples],
		"parameters": parameters,
		"load_seconds": round(load_seconds, 3),
		"timing": timing_report(warm_times, measured_times),
		"evaluation": {
			"examples": len(evaluation_batches),
			"seconds": round(evaluation_seconds, 4),
			evaluation_name: round(evaluation_value, 6),
		},
		"checkpoint": {
			"size_mb": checkpoint_size,
			"save_seconds": round(checkpoint_save_seconds, 3),
		},
		"memory": memory.report(),
	}
	del models, optimizer, batches, evaluation_batches
	gc.collect()
	return result


def run_all(output_path: Path) -> None:
	stage_results = {}
	for stage in ("cpt", "sft", "dpo"):
		stage_path = output_path.with_name(f"{output_path.stem}-{stage}.json")
		print(f"\n=== Starting isolated {stage.upper()} calibration ===", flush=True)
		subprocess.run(
			[sys.executable, str(Path(__file__).resolve()), "--stage", stage, "--output", str(stage_path)],
			check=True,
		)
		stage_results[stage] = json.loads(stage_path.read_text(encoding="utf-8"))
		stage_path.unlink()
	report = {
		"method": {
			"description": "Three warm-up updates followed by five measured updates in a fresh process per objective.",
			"batch_size": 1,
			"device": "cpu",
			"seed": SEED,
			"corpus": "The Weight of Distant Light (36 train chapters, 4 held out)",
		},
		"hardware": hardware_report(),
		"stages": stage_results,
	}
	output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print(f"\nCalibration report: {output_path}")
	for stage, result in stage_results.items():
		timing = result["timing"]
		print(
			f"{stage.upper():>3}: {timing['mean_seconds_per_step']:.3f}s/step | "
			f"peak RSS {result['memory']['peak_rss_mb']:.1f} MB | "
			f"checkpoint {result['checkpoint']['size_mb']:.1f} MB"
		)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--stage", choices=("all", "cpt", "sft", "dpo"), default="all")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	args = parser.parse_args()
	args.output.parent.mkdir(parents=True, exist_ok=True)
	if args.stage == "all":
		run_all(args.output.resolve())
	else:
		result = run_stage(args.stage)
		args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
		print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()

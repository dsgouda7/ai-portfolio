"""Real CPU generation and library metrics for the fixed Riverside teaching set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

from bert_score import BERTScorer
from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
import nltk
from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation_shared import RIVERSIDE_CASES, SEED, tokenize


HERE = Path(__file__).resolve().parent
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ENCODER_ID = "distilbert/distilbert-base-uncased"
ENCODER_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
GENERATION = {"max_new_tokens": 64, "do_sample": False, "use_cache": True}
SYSTEM_PROMPT = (
    "Answer the question using only the supplied context. Use one short sentence. "
    "If the context does not answer it, say that you cannot tell."
)
RESULT_PATH = HERE / "results" / "metrics-cpu-smollm2-135m.json"
ASSET_PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors",
    "model.safetensors.index.json", "model-*.safetensors",
    "tokenizer*", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "vocab.txt", "merges.txt",
]


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_checkpoint_inventory(root: Path | None = None) -> list[dict]:
    """Inspect saved HF packages/adapters, not arbitrary PyTorch state dictionaries."""
    root = root if root is not None else HERE.parent / "06-llm-finetuning" / "checkpoints"
    configs = sorted(set(root.rglob("config.json")) | set(root.rglob("adapter_config.json")))
    rows = []
    for config_path in configs:
        path = config_path.parent
        adapter = config_path.name == "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        weights = list(path.glob("adapter_model*.safetensors" if adapter else "model*.safetensors"))
        tokenizer_saved = (path / "tokenizer_config.json").is_file()
        base = config.get("base_model_name_or_path") if adapter else None
        ready = bool(weights) and (tokenizer_saved or bool(base))
        rows.append({
            "checkpoint": str(path.relative_to(root)), "kind": "adapter" if adapter else "full",
            "ready": ready, "tokenizer_saved": tokenizer_saved, "base_model": base,
            "note": (
                "Load explicitly; availability does not establish training/test independence."
                if ready else "Needs weights and identifiable tokenizer; not silently substituted."
            ),
        })
    return rows


def model_options() -> list[dict]:
    cached = try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    local = local_checkpoint_inventory()
    local_rows = [
        {"source": "local", "model": row["checkpoint"], "ready": row["ready"], "note": row["note"]}
        for row in local
    ] or [{"source": "local", "model": "No saved HF checkpoint found", "ready": False,
           "note": "The HF source is selected explicitly, not used as an error fallback."}]
    return [
        {"source": "hf", "model": MODEL_ID, "ready": True,
         "note": "Pinned weights cached." if isinstance(cached, str) else "Download pinned weights on first use."},
        *local_rows,
    ]


def download_snapshot(model_id: str, revision: str) -> Path:
    return Path(snapshot_download(model_id, revision=revision, allow_patterns=ASSET_PATTERNS))


def load_eval_model(source: str = "hf", local_path: Path | None = None):
    """An explicitly selected broken local checkpoint is an error, never an HF fallback."""
    if source == "hf":
        if local_path is not None:
            raise ValueError("Choose source='local' when supplying a local checkpoint.")
        path = download_snapshot(MODEL_ID, MODEL_REVISION)
        provenance = {"source": "hf", "model_id": MODEL_ID, "revision": MODEL_REVISION}
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, local_files_only=True, trust_remote_code=False,
        )
        tokenizer_path = path
    elif source == "local":
        if local_path is None or not local_path.is_dir():
            raise FileNotFoundError("source='local' requires an existing checkpoint directory.")
        path = local_path.resolve()
        adapter_path = path / "adapter_config.json"
        if adapter_path.is_file():
            from peft import PeftModel

            config = json.loads(adapter_path.read_text(encoding="utf-8"))
            base_id = config["base_model_name_or_path"]
            revision = config.get("revision") or HfApi().model_info(base_id).sha
            if not revision:
                raise ValueError("The adapter's base revision could not be resolved.")
            base_path = download_snapshot(base_id, revision)
            base = AutoModelForCausalLM.from_pretrained(
                base_path, dtype=torch.float32, local_files_only=True, trust_remote_code=False,
            )
            model = PeftModel.from_pretrained(base, path, local_files_only=True)
            tokenizer_path = path if (path / "tokenizer_config.json").is_file() else base_path
            identity = {"base_model_id": base_id, "base_revision": revision}
        else:
            if not (path / "tokenizer_config.json").is_file():
                raise ValueError("The full checkpoint has no saved tokenizer; supply a complete HF package.")
            model = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float32, local_files_only=True, trust_remote_code=False,
            )
            tokenizer_path = path
            identity = {}
        files = sorted(path.glob("*.json")) + sorted(path.glob("*.safetensors"))
        provenance = {
            "source": "local", "checkpoint_name": path.name, **identity,
            "artifact_sha256": {item.name: file_sha256(item) for item in files},
            "independence": "Not established: inspect the earlier training lineage before quality comparisons.",
        }
    else:
        raise ValueError("source must be 'hf' or 'local'.")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False,
    )
    if tokenizer.chat_template is None:
        raise ValueError("This contextual-answer experiment requires an instruction model with a chat template.")
    model.to("cpu").eval()
    return model, tokenizer, provenance


def prompt_for(case: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {case['context']}\nQuestion: {case['query']}"},
    ], tokenize=False, add_generation_prompt=True)


def reference_perplexity(model, prompt_ids: torch.Tensor, reference_ids: torch.Tensor) -> dict:
    """Score only the fixed answer tokens, not the prompt or the model's own guesses."""
    if prompt_ids.shape[1] == 0 or reference_ids.shape[1] == 0:
        raise ValueError("Prompt and reference must both contain tokens.")
    joined = torch.cat((prompt_ids, reference_ids), dim=1)
    labels = joined.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    with torch.inference_mode():
        loss = model(input_ids=joined, labels=labels, use_cache=False).loss.item()
    count = reference_ids.shape[1]
    return {"reference_nll_sum": loss * count, "reference_tokens": count,
            "reference_ppl": math.exp(loss)}


def generate_cases(model, tokenizer, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        prompt = prompt_for(case, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        reference_ids = tokenizer(
            case["reference"], return_tensors="pt", add_special_tokens=False,
        ).input_ids
        needed = inputs.input_ids.shape[1] + max(reference_ids.shape[1], GENERATION["max_new_tokens"])
        if needed > model.config.max_position_embeddings:
            raise ValueError(f"{case['id']}: context would exceed the model limit; nothing was truncated.")
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, **GENERATION, pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output[0, inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if not answer:
            raise ValueError(f"{case['id']}: the model generated no answer; inspect the run.")
        rows.append({
            "id": case["id"], "slice": case["slice"], "query": case["query"],
            "context": case["context"], "reference": case["reference"],
            "prompt": prompt, "answer": answer, "generated_tokens": len(new_ids),
            "generation_seconds": time.perf_counter() - started,
            **reference_perplexity(model, inputs.input_ids, reference_ids),
        })
    return rows


def prepare_wordnet() -> str:
    try:
        wordnet.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True, raise_on_error=True)
        wordnet.ensure_loaded()
    return str(wordnet.get_version())


class LessonTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return tokenize(text)


def score_answers(rows: list[dict], encoder_path: Path) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError("Supply at least one measured answer/reference pair.")
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], tokenizer=LessonTokenizer(), use_stemmer=False,
    )
    # Swap "rougeL" for "rouge1"/"rouge2"; the answer/reference loop stays identical.
    # BLEU-1/2/3 change max_ngram_order, not the loop. Report that setting with the score.
    bleu = BLEU(tokenize="none", smooth_method="exp", effective_order=True)
    bert = BERTScorer(
        model_type=str(encoder_path), num_layers=6, device="cpu", batch_size=4,
        nthreads=4, idf=False, rescale_with_baseline=False, use_fast_tokenizer=True,
    )
    predictions = [row["answer"] for row in rows]
    references = [row["reference"] for row in rows]
    bert_p, bert_r, bert_f1 = bert.score(predictions, references)
    scored = []
    for index, row in enumerate(rows):
        prediction, reference = predictions[index], references[index]
        scores = rouge.score(reference, prediction)
        candidate_terms, reference_terms = tokenize(prediction), tokenize(reference)
        scored.append({
            **row,
            **{f"{name}_{part}": getattr(value, part)
               for name, value in scores.items() for part in ("precision", "recall", "fmeasure")},
            "bleu_0_100": bleu.sentence_score(" ".join(candidate_terms), [" ".join(reference_terms)]).score,
            # METEOR keeps library defaults; alpha/beta/gamma change the recall/order trade-off.
            "meteor": meteor_score([reference_terms], candidate_terms),
            # Precision/recall/F1 reuse the same BERTScore matches; these are not three new models.
            "bertscore_precision": bert_p[index].item(), "bertscore_recall": bert_r[index].item(),
            "bertscore_f1": bert_f1[index].item(),
        })
    metric_names = [key for key in scored[0] if key.startswith(("rouge", "meteor", "bertscore"))]
    summary = {key: sum(row[key] for row in scored) / len(scored) for key in metric_names}
    summary["bleu_0_100"] = bleu.corpus_score(
        [" ".join(tokenize(text)) for text in predictions],
        [[" ".join(tokenize(text)) for text in references]],
    ).score
    # Corpus BLEU is not mean sentence BLEU. Corpus perplexity is token-weighted, not mean PPL.
    summary["reference_tokens"] = sum(row["reference_tokens"] for row in scored)
    summary["reference_nll_sum"] = sum(row["reference_nll_sum"] for row in scored)
    summary["reference_ppl"] = math.exp(summary["reference_nll_sum"] / summary["reference_tokens"])
    summary["bleu_signature"] = str(bleu.get_signature())
    return scored, summary


def run_evaluation(source: str = "hf", local_path: Path | None = None) -> dict:
    previous_threads = torch.get_num_threads()
    started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            options = model_options()
            print("Available sources:", json.dumps(options), flush=True)
            model, tokenizer, provenance = load_eval_model(source, local_path)
            rows = generate_cases(model, tokenizer, RIVERSIDE_CASES)
            del model
            wordnet_version = prepare_wordnet()
            encoder_path = download_snapshot(ENCODER_ID, ENCODER_REVISION)
            scored, summary = score_answers(rows, encoder_path)
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "schema_version": 1, "evidence": "Local measured; four visible teaching cases, not a sealed quality benchmark.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": fingerprint(RIVERSIDE_CASES), "test_ids": [case["id"] for case in RIVERSIDE_CASES],
        "model": provenance, "available_sources": options,
        "generation": GENERATION, "system_prompt": SYSTEM_PROMPT, "seed": SEED,
        "scorers": {
            "rouge": "ROUGE-1/2/L precision/recall/F1; shared lowercase regex tokenizer; no stemming",
            "bleu": summary["bleu_signature"],
            "meteor": {"alpha": 0.9, "beta": 3.0, "gamma": 0.5, "wordnet_version": wordnet_version},
            "bertscore": {"model_id": ENCODER_ID, "revision": ENCODER_REVISION,
                          "layers": 6, "idf": False, "rescale_with_baseline": False},
            "perplexity": "Fixed reference only, conditioned on the exact chat prompt; no EOS target; token-weighted.",
        },
        "runtime": {"device": "cpu", "dtype": "float32", "threads": 4,
                    "seconds": time.perf_counter() - started, "python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in (
                        "torch", "transformers", "huggingface-hub", "rouge-score", "sacrebleu", "nltk", "bert-score",
                    )}},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_code_sha256": hashlib.sha256((HERE / "evaluation_shared.py").read_bytes()).hexdigest(),
        "items": scored, "aggregate": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("hf", "local"), default="hf")
    parser.add_argument("--local-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_evaluation(args.source, args.local_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print("Saved measured answers, scores and provenance:", args.output)

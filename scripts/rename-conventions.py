#!/usr/bin/env python3
"""
Rename script for ai-portfolio:
  - Directories  -> snake_case (Python convention)
  - .md files    -> kebab-case  (GitHub convention)

Exceptions: README.md, CONTRIBUTING.md, AGENTS.md are kept as-is.

Also updates all internal markdown and YAML links/references.

Usage:
    python scripts/rename_conventions.py           # execute renames
    python scripts/rename_conventions.py --dry-run # preview only
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
DRY_RUN = "--dry-run" in sys.argv

KEEP_FILES = {"README.md", "CONTRIBUTING.md", "AGENTS.md"}
SKIP_DIRS = {".venv", ".git", ".kilo", "__pycache__", ".mkdocs-site"}

# ─── Directory segment renames (old name -> new name) ────────────────────────

DIR_SEG_MAP: dict[str, str] = {
    # top-level notes tracks
    "AI": "ai",
    "AIInfrastructure": "ai_infrastructure",
    "Archived": "archived",
    "InterviewGuides": "interview_guides",
    "MathUnderTheHood": "math_under_the_hood",
    "ML": "ml",
    "MultiAgentAI": "multi_agent_ai",
    "MultimodalAI": "multimodal_ai",
    # AI sub-chapters
    "CostAndLatency": "cost_and_latency",
    "CoTReasoning": "cot_reasoning",
    "EvaluatingAISystems": "evaluating_ai_systems",
    "FineTuning": "fine_tuning",
    "LLMFundamentals": "llm_fundamentals",
    "PromptEngineering": "prompt_engineering",
    "RAGAndEmbeddings": "rag_and_embeddings",
    "ReActAndSemanticKernel": "react_and_semantic_kernel",
    "SafetyAndHallucination": "safety_and_hallucination",
    "VectorDBs": "vector_dbs",
    # AIInfrastructure sub-chapters
    "GPUArchitecture": "gpu_architecture",
    "InferenceOptimization": "inference_optimization",
    "MemoryAndComputeBudgets": "memory_and_compute_budgets",
    "ParallelismAndDistributedTraining": "parallelism_and_distributed_training",
    "QuantizationAndPrecision": "quantization_and_precision",
    # ML top-level sections
    "01-Regression": "01_regression",
    "02-Classification": "02_classification",
    "03-NeuralNetworks": "03_neural_networks",
    "04-RecommenderSystems": "04_recommender_systems",
    "05-AnomalyDetection": "05_anomaly_detection",
    "06-ReinforcementLearning": "06_reinforcement_learning",
    "07-UnsupervisedLearning": "07_unsupervised_learning",
    "08-EnsembleMethods": "08_ensemble_methods",
    # ML/01-Regression chapters
    "ch01-linear-regression": "ch01_linear_regression",
    "ch02-multiple-regression": "ch02_multiple_regression",
    "ch03-feature-importance": "ch03_feature_importance",
    "ch04-polynomial-features": "ch04_polynomial_features",
    "ch05-regularization": "ch05_regularization",
    "ch06-metrics": "ch06_metrics",
    "ch07-hyperparameter-tuning": "ch07_hyperparameter_tuning",
    # ML/02-Classification chapters
    "ch01-logistic-regression": "ch01_logistic_regression",
    "ch02-classical-classifiers": "ch02_classical_classifiers",
    "ch03-metrics": "ch03_metrics",
    "ch04-svm": "ch04_svm",
    "ch05-hyperparameter-tuning": "ch05_hyperparameter_tuning",
    # ML/03-NeuralNetworks chapters
    "ch01-xor-problem": "ch01_xor_problem",
    "ch02-neural-networks": "ch02_neural_networks",
    "ch03-backprop-optimisers": "ch03_backprop_optimisers",
    "ch04-regularisation": "ch04_regularisation",
    "ch05-cnns": "ch05_cnns",
    "ch06-rnns-lstms": "ch06_rnns_lstms",
    "ch07-mle-loss-functions": "ch07_mle_loss_functions",
    "ch08-tensorboard": "ch08_tensorboard",
    "ch09-sequences-to-attention": "ch09_sequences_to_attention",
    "ch10-transformers": "ch10_transformers",
    # ML/04-RecommenderSystems chapters
    "ch01-fundamentals": "ch01_fundamentals",
    "ch02-collaborative-filtering": "ch02_collaborative_filtering",
    "ch03-matrix-factorization": "ch03_matrix_factorization",
    "ch04-neural-cf": "ch04_neural_cf",
    "ch05-hybrid-systems": "ch05_hybrid_systems",
    "ch06-cold-start-production": "ch06_cold_start_production",
    # ML/05-AnomalyDetection chapters
    "ch01-statistical-methods": "ch01_statistical_methods",
    "ch02-isolation-forest": "ch02_isolation_forest",
    "ch03-autoencoders": "ch03_autoencoders",
    "ch04-one-class-svm": "ch04_one_class_svm",
    "ch05-ensemble-anomaly": "ch05_ensemble_anomaly",
    "ch06-production": "ch06_production",
    # ML/06-ReinforcementLearning chapters
    "ch01-mdps": "ch01_mdps",
    "ch02-dynamic-programming": "ch02_dynamic_programming",
    "ch03-q-learning": "ch03_q_learning",
    "ch04-dqn": "ch04_dqn",
    "ch05-policy-gradients": "ch05_policy_gradients",
    "ch06-modern-rl": "ch06_modern_rl",
    # ML/07-UnsupervisedLearning chapters
    "ch01-clustering": "ch01_clustering",
    "ch02-dimensionality-reduction": "ch02_dimensionality_reduction",
    "ch03-unsupervised-metrics": "ch03_unsupervised_metrics",
    # ML/08-EnsembleMethods chapters
    "ch01-ensembles": "ch01_ensembles",
    "ch02-boosting": "ch02_boosting",
    "ch03-xgboost-lightgbm": "ch03_xgboost_lightgbm",
    "ch04-shap": "ch04_shap",
    "ch05-stacking": "ch05_stacking",
    # MathUnderTheHood chapters
    "ch01-linear-algebra": "ch01_linear_algebra",
    "ch02-nonlinear-algebra": "ch02_nonlinear_algebra",
    "ch03-calculus-intro": "ch03_calculus_intro",
    "ch04-small-steps": "ch04_small_steps",
    "ch05-matrices": "ch05_matrices",
    "ch06-gradient-chain-rule": "ch06_gradient_chain_rule",
    "ch07-probability-statistics": "ch07_probability_statistics",
    # MultiAgentAI sub-chapters
    "A2A": "a2a",
    "AgentFrameworks": "agent_frameworks",
    "EventDrivenAgents": "event_driven_agents",
    "MCP": "mcp",
    "MessageFormats": "message_formats",
    "SharedMemory": "shared_memory",
    "TrustAndSandboxing": "trust_and_sandboxing",
    # MultimodalAI sub-chapters
    "AudioGeneration": "audio_generation",
    "CLIP": "clip",
    "DiffusionModels": "diffusion_models",
    "GenerativeEvaluation": "generative_evaluation",
    "GuidanceConditioning": "guidance_conditioning",
    "LatentDiffusion": "latent_diffusion",
    "LocalDiffusionLab": "local_diffusion_lab",
    "MultimodalFoundations": "multimodal_foundations",
    "MultimodalLLMs": "multimodal_llms",
    "Schedulers": "schedulers",
    "TextToImage": "text_to_image",
    "TextToVideo": "text_to_video",
    "VisionTransformers": "vision_transformers",
    # projects
    "rag-pipeline": "rag_pipeline",
    "linear-regression": "linear_regression",
    # Archived
    "Chronicles": "chronicles",
}

# ─── File renames (.md only, except KEEP_FILES) ──────────────────────────────

FILE_NAME_MAP: dict[str, str] = {
    "AIPrimer.md": "ai-primer.md",
    "AUTHORING_GUIDE.md": "authoring-guide.md",
    "CostAndLatency.md": "cost-and-latency.md",
    "CoTReasoning_Supplement.md": "cot-reasoning-supplement.md",
    "CoTReasoning.md": "cot-reasoning.md",
    "EvaluatingAISystems.md": "evaluating-ai-systems.md",
    "FineTuning.md": "fine-tuning.md",
    "LLMFundamentals.md": "llm-fundamentals.md",
    "PromptEngineering.md": "prompt-engineering.md",
    "RAGAndEmbeddings_Supplement.md": "rag-and-embeddings-supplement.md",
    "RAGAndEmbeddings.md": "rag-and-embeddings.md",
    "ReActAndSemanticKernel_Supplement.md": "react-and-semantic-kernel-supplement.md",
    "ReActAndSemanticKernel.md": "react-and-semantic-kernel.md",
    "SafetyAndHallucination.md": "safety-and-hallucination.md",
    "VectorDBs_Supplement.md": "vector-dbs-supplement.md",
    "VectorDBs.md": "vector-dbs.md",
    "GPUArchitecture.md": "gpu-architecture.md",
    "InferenceOptimization.md": "inference-optimization.md",
    "MemoryBudgets.md": "memory-budgets.md",
    "Parallelism.md": "parallelism.md",
    "Quantization.md": "quantization.md",
    "authoring_guidelines.md": "authoring-guidelines.md",
    "AgenticAI.md": "agentic-ai.md",
    "AIInfrastructure.md": "ai-infrastructure.md",
    "MultiAgentAI.md": "multi-agent-ai.md",
    "MultimodalAI.md": "multimodal-ai.md",
    "content_audit.md": "content-audit.md",
    "Interview_guide.md": "interview-guide.md",
    "validation_report.md": "validation-report.md",
    "ANIMATION_PLAN.md": "animation-plan.md",
    "CLIP.md": "clip.md",
    "DiffusionModels.md": "diffusion-models.md",
    "GenerativeEvaluation.md": "generative-evaluation.md",
    "GuidanceConditioning.md": "guidance-conditioning.md",
    "LatentDiffusion.md": "latent-diffusion.md",
    "LocalDiffusionLab.md": "local-diffusion-lab.md",
    "MultimodalFoundations.md": "multimodal-foundations.md",
    "MultimodalLLMs.md": "multimodal-llms.md",
    "Schedulers.md": "schedulers.md",
    "TextToImage.md": "text-to-image.md",
    "TextToVideo.md": "text-to-video.md",
    "VisionTransformers.md": "vision-transformers.md",
    "GRAND_CHALLENGE.md": "grand-challenge.md",
    "plan.md": "plan.md",  # already fine
}

# ─── Path transformation helpers ─────────────────────────────────────────────


def transform_rel_path(rel: str) -> str:
    """
    Given a path relative to ROOT (using forward slashes), return the
    transformed path after applying all directory and file renames.
    """
    parts = rel.replace("\\", "/").split("/")
    new_parts: list[str] = []
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        if is_last and "." in part:
            # It is a file
            if part not in KEEP_FILES:
                new_parts.append(FILE_NAME_MAP.get(part, part))
            else:
                new_parts.append(part)
        else:
            new_parts.append(DIR_SEG_MAP.get(part, part))
    return "/".join(new_parts)


def resolve_link(link: str, source_file: Path) -> str:
    """
    Given a relative link found inside `source_file`, return the
    updated link pointing to the renamed target.

    External links, anchors, and non-.md targets that don't appear
    in our mapping are returned unchanged.
    """
    # Keep external / anchor links unchanged
    if re.match(r"^(https?://|mailto:|#)", link):
        return link

    # Split off any fragment
    fragment = ""
    if "#" in link:
        path_part, fragment = link.split("#", 1)
        fragment = "#" + fragment
    else:
        path_part = link

    if not path_part:
        return link  # pure fragment

    # Resolve the path to an absolute path (old location)
    try:
        source_dir = source_file.parent
        abs_target = (source_dir / path_part).resolve()
        # Must be inside repo
        rel_target = abs_target.relative_to(ROOT)
    except (ValueError, OSError):
        # Can't resolve — apply segment transforms naively
        new_path = transform_rel_path(path_part.replace("\\", "/"))
        return new_path + fragment

    # Transform the target path
    rel_target_str = str(rel_target).replace("\\", "/")
    new_rel_target_str = transform_rel_path(rel_target_str)

    # Compute the new source directory (after its own renames)
    old_source_rel = str(source_file.relative_to(ROOT)).replace("\\", "/")
    new_source_rel = transform_rel_path(old_source_rel)
    new_source_dir = (ROOT / new_source_rel).parent

    new_abs_target = ROOT / new_rel_target_str

    # Compute new relative path from new source dir to new target
    try:
        new_rel = os.path.relpath(new_abs_target, new_source_dir)
        new_rel = new_rel.replace("\\", "/")
    except ValueError:
        new_rel = new_rel_target_str

    return new_rel + fragment


# ─── Link updater ─────────────────────────────────────────────────────────────

# Matches [text](url) and ![alt](url)
_LINK_RE = re.compile(r"(!?\[[^\]]*\])\(([^)]+)\)")


def update_links_in_text(content: str, source_file: Path) -> str:
    def replacer(m: re.Match) -> str:
        prefix = m.group(1)  # [text] or ![alt]
        raw_link = m.group(2)

        # Preserve leading/trailing whitespace inside parens
        stripped = raw_link.strip()
        leading = raw_link[: len(raw_link) - len(raw_link.lstrip())]
        trailing = raw_link[len(raw_link.rstrip()) :]

        new_link = resolve_link(stripped, source_file)
        return f"{prefix}({leading}{new_link}{trailing})"

    return _LINK_RE.sub(replacer, content)


# ─── Collection phase ─────────────────────────────────────────────────────────


def collect_renames():
    """
    Walk the repo and collect:
      - dir_renames: list of (old_abs_path, new_abs_path) for directories
      - file_renames: list of (old_abs_path, new_abs_path) for .md files
    Both lists exclude items with unchanged names.
    """
    dir_renames: list[tuple[Path, Path]] = []
    file_renames: list[tuple[Path, Path]] = []

    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Prune dirs we never want to enter
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        dir_path = Path(dirpath)

        # Check each subdirectory name
        for dirname in list(dirnames):
            new_dirname = DIR_SEG_MAP.get(dirname, dirname)
            if new_dirname != dirname:
                old_dir = dir_path / dirname
                # Build new absolute path by transforming all path segments
                old_rel = str(old_dir.relative_to(ROOT)).replace("\\", "/")
                new_rel = transform_rel_path(old_rel + "/placeholder")[: -len("/placeholder")]
                new_dir = ROOT / new_rel
                dir_renames.append((old_dir, new_dir))

        # Check each file
        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            if filename in KEEP_FILES:
                continue
            new_filename = FILE_NAME_MAP.get(filename, filename)
            if new_filename != filename:
                old_file = dir_path / filename
                file_renames.append((old_file, dir_path / new_filename))

    return dir_renames, file_renames


# ─── Text-file link updater ───────────────────────────────────────────────────

TEXT_EXTENSIONS = {".md", ".yml", ".yaml", ".agent.md"}


def update_all_links(dir_renames: list, file_renames: list):
    """
    Update all markdown links in .md and .yml files throughout the repo.
    Must be called BEFORE the actual renames take place.
    """
    updated_count = 0

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            if not any(filename.endswith(ext) for ext in (".md", ".yml", ".yaml")):
                continue

            file_path = Path(dirpath) / filename
            try:
                original = file_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"  WARN: cannot read {file_path}: {e}")
                continue

            updated = update_links_in_text(original, file_path)

            if updated != original:
                updated_count += 1
                if DRY_RUN:
                    rel = file_path.relative_to(ROOT)
                    print(f"  [links] would update: {rel}")
                else:
                    file_path.write_text(updated, encoding="utf-8")
                    rel = file_path.relative_to(ROOT)
                    print(f"  [links] updated: {rel}")

    return updated_count


# ─── Rename executor ──────────────────────────────────────────────────────────


def _current_path(orig: Path, applied: list[tuple[Path, Path]]) -> Path:
    """
    Given the original path of a dir/file, compute its current location
    after all previously-applied parent renames.
    Iterates until no more transforms apply.
    Uses str comparison (case-sensitive) to correctly track case-only renames.
    """
    current = orig
    changed = True
    while changed:
        changed = False
        for done_old, done_new in applied:
            try:
                rel = current.relative_to(done_old)
                candidate = done_new / rel
                # Use str comparison so case-only changes (ML->ml) are detected
                if str(candidate) != str(current):
                    current = candidate
                    changed = True
                    break
            except ValueError:
                pass
    return current


def _rename_with_retry(src: Path, dst: Path) -> None:
    """
    Rename src to dst.
    Strategy on PermissionError (VS Code / Windows file-watcher lock):
      1. One Python attempt.
      2. Immediate PowerShell Move-Item fallback (uses copy+delete on Windows
         when the directory cannot be renamed atomically).
      3. Ten more Python retries with 1-second delays as final backstop.
    """
    # Attempt 1: standard Python rename
    try:
        src.rename(dst)
        return
    except PermissionError:
        pass

    # Attempt 2: PowerShell Move-Item (handles locked dirs via copy+delete)
    ps_cmd = f"Move-Item -Path '{src}' -Destination '{dst}' -Force"
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and dst.exists():
            return
    except Exception:
        pass

    # Attempts 3–12: wait for file-watcher to release, retry Python
    for attempt in range(10):
        time.sleep(1.0)
        try:
            src.rename(dst)
            return
        except PermissionError:
            pass

    raise PermissionError(f"Cannot rename after all attempts: {src} -> {dst}")


def _safe_rename_dir(src: Path, dst: Path) -> bool:
    """
    Rename src to dst.  Returns True on success, False if skipped.
    - Case-only renames (ML -> ml) use a temp intermediate on Windows NTFS.
    - If dst already exists as an empty directory (e.g. created by a prior
      partial run), remove it first so the rename can proceed.
    - Retries on transient PermissionError from file watchers.
    """
    if str(src).lower() == str(dst).lower():
        # Case-only rename: go via a temp name (needed on case-insensitive NTFS)
        tmp = src.parent / (src.name + "_RENAME_TMP_")
        try:
            _rename_with_retry(src, tmp)
        except PermissionError:
            print(f"    WARN: skipping case-only rename (permission denied): {src.name}")
            return False
        _rename_with_retry(tmp, dst)
        return True
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # On Windows, os.rename fails if dst exists, even when empty.
        if dst.exists():
            if dst.is_dir() and not any(dst.iterdir()):
                dst.rmdir()
            else:
                raise FileExistsError(
                    f"Cannot rename {src} -> {dst}: destination already exists"
                    " and is not empty."
                )
        _rename_with_retry(src, dst)
        return True


def execute_renames(dir_renames: list, file_renames: list):
    """
    Rename directories top-down (shallowest first) then files.
    Tracks applied renames to resolve current filesystem paths of
    items whose ancestors were already renamed.
    """
    # Sort shallowest first (fewest path separators = higher in tree)
    dir_renames_sorted = sorted(
        dir_renames, key=lambda t: str(t[0]).count(os.sep)
    )

    # Each entry: (current_path_at_rename_time, target_new)
    # Storing current (not orig) so subsequent chained lookups work correctly.
    applied: list[tuple[Path, Path]] = []

    dir_count = 0
    for orig_old, target_new in dir_renames_sorted:
        current = _current_path(orig_old, applied)

        # Use str comparison: case-only renames (ML->ml) must not be skipped
        if str(current) == str(target_new):
            applied.append((current, target_new))
            continue

        rel_cur = str(current.relative_to(ROOT))
        rel_new = str(target_new.relative_to(ROOT))

        if DRY_RUN:
            print(f"  [dir]  {rel_cur}  ->  {rel_new}")
        else:
            if not current.exists():
                print(f"  WARN: dir not found: {current}")
                applied.append((current, target_new))
                continue
            renamed = _safe_rename_dir(current, target_new)
            if renamed:
                print(f"  [dir]  {rel_cur}  ->  {rel_new}")
            else:
                # Skipped (e.g. case-only on locked dir); still record transform
                print(f"  [skip] {rel_cur}  ->  {rel_new}  (case-only, locked)")

        applied.append((current, target_new))
        dir_count += 1

    # After directory renames, files are in renamed dirs with old filenames.
    file_count = 0
    for orig_old_file, _ in file_renames:
        current_file = _current_path(orig_old_file, applied)

        old_filename = orig_old_file.name
        new_filename = FILE_NAME_MAP.get(old_filename, old_filename)
        if new_filename == old_filename:
            continue

        new_file_path = current_file.parent / new_filename

        # Use str comparison: case-only file renames must not be skipped
        if str(current_file) == str(new_file_path):
            continue

        rel_cur = str(current_file.relative_to(ROOT))
        rel_new = str(new_file_path.relative_to(ROOT))

        if DRY_RUN:
            print(f"  [file] {rel_cur}  ->  {rel_new}")
        else:
            if not current_file.exists():
                print(f"  WARN: file not found: {current_file}")
                continue
            current_file.rename(new_file_path)
            print(f"  [file] {rel_cur}  ->  {rel_new}")

        file_count += 1

    return dir_count, file_count


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    mode = "DRY RUN" if DRY_RUN else "EXECUTE"
    print(f"\n{'='*60}")
    print(f"  rename_conventions.py  [{mode}]")
    print(f"  Root: {ROOT}")
    print(f"{'='*60}\n")

    print("Collecting renames...")
    dir_renames, file_renames = collect_renames()
    print(f"  {len(dir_renames)} directory renames")
    print(f"  {len(file_renames)} file renames\n")

    print("Updating internal links...")
    link_count = update_all_links(dir_renames, file_renames)
    print(f"  {link_count} files had links updated\n")

    print("Renaming directories and files...")
    dir_count, file_count = execute_renames(dir_renames, file_renames)
    print(f"\n  Done: {dir_count} directories renamed, {file_count} files renamed")

    if DRY_RUN:
        print("\n  (dry run — no changes written)\n")
    else:
        print("\n  All renames complete.\n")


if __name__ == "__main__":
    main()

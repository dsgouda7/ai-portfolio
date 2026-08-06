from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import CompatibilityError
from .models import ChatCompletionRequest, VerifiedRelease


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class TransformersPeftBackend:
    """Lazy SmolLM2 causal-LM backend with a verified PEFT adapter."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        base_model_local_only: bool = False,
        base_model_directory: Path | None = None,
    ) -> None:
        self._device = device
        self._base_model_local_only = base_model_local_only
        self._base_model_directory = base_model_directory
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self, release: VerifiedRelease) -> None:
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")

        precision_types = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if release.manifest.precision not in precision_types:
            raise CompatibilityError("configured backend does not support quantized precision")

        tokenizer_directory = Path(release.paths.tokenizer).parent
        adapter_directory = Path(release.paths.adapter).parent
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(tokenizer_directory),
            local_files_only=True,
            trust_remote_code=False,
        )
        base_model_source = str(self._base_model_directory or release.manifest.base_model.id)
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_source,
            revision=None if self._base_model_directory else release.manifest.base_model.revision,
            local_files_only=self._base_model_local_only,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=precision_types[release.manifest.precision],
        )
        model = peft.PeftModel.from_pretrained(
            base_model,
            str(adapter_directory),
            is_trainable=False,
        )
        model.to(self._device)
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self._tokenizer = tokenizer
        self._model = model

    def warmup(self) -> None:
        self._generate(
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )

    def count_input_tokens(self, request: ChatCompletionRequest) -> int:
        tokenizer = self._require_tokenizer()
        prompt = self._render_prompt(request)
        encoded = tokenizer(prompt, add_special_tokens=False)
        return len(encoded["input_ids"])

    def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        return self._generate(
            messages=[message.model_dump(exclude_none=True) for message in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

    def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
    ) -> GenerationResult:
        tokenizer = self._require_tokenizer()
        model = self._require_model()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = encoded.to(self._device)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        do_sample = temperature > 0
        generation_options: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_options.update(temperature=temperature, top_p=top_p)
        output = model.generate(**encoded, **generation_options)
        generated_ids = output[0, prompt_tokens:]
        completion_tokens = int(generated_ids.shape[-1])
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = self._apply_stop(text, stop)
        finish_reason = "length" if completion_tokens >= max_tokens else "stop"
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def _render_prompt(self, request: ChatCompletionRequest) -> str:
        tokenizer = self._require_tokenizer()
        messages = [message.model_dump(exclude_none=True) for message in request.messages]
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("model backend has not been loaded")
        return self._model

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("tokenizer backend has not been loaded")
        return self._tokenizer

    @staticmethod
    def _apply_stop(text: str, stop: str | list[str] | None) -> str:
        sequences = [stop] if isinstance(stop, str) else stop or []
        positions = [position for value in sequences if (position := text.find(value)) >= 0]
        return text[: min(positions)] if positions else text

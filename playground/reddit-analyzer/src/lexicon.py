from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Set

from datasets import load_dataset


BASE_NEGATIVE_LEXICON: List[str] = [
    "idiot",
    "stupid",
    "trash",
    "garbage",
    "terrorist",
    "vermin",
    "subhuman",
    "criminal",
    "thug",
    "degenerate",
    "scum",
    "parasite",
    "filthy",
    "hate",
    "worthless",
    "backward",
    "disgusting",
    "invader",
    "inferior",
    "savage",
]


@dataclass
class LexiconManager:
    base_terms: Iterable[str] = field(default_factory=lambda: BASE_NEGATIVE_LEXICON)

    def get_base_lexicon(self) -> Set[str]:
        return {t.strip().lower() for t in self.base_terms if t and t.strip()}

    def load_hf_seed_lexicon(
        self,
        dataset_name: str = "SEACrowd/tgl_profanity",
        split: str = "train",
        text_columns: Iterable[str] = ("text", "word", "term", "profanity"),
        max_rows: int = 20000,
    ) -> Set[str]:
        """Load profanity/abuse terms from a Hugging Face dataset.

        The loader is schema-flexible and scans likely text columns.
        """
        ds = load_dataset(dataset_name, split=split)
        terms: Set[str] = set()

        for idx, row in enumerate(ds):
            if idx >= max_rows:
                break
            for col in text_columns:
                value = row.get(col)
                if isinstance(value, str) and value.strip():
                    term = value.strip().lower()
                    if len(term) <= 64 and " " not in term:
                        terms.add(term)

        return terms

    def build_lexicon(
        self,
        include_hf_seed: bool = True,
        dataset_name: str = "SEACrowd/tgl_profanity",
    ) -> Set[str]:
        lexicon = set(self.get_base_lexicon())
        if include_hf_seed:
            try:
                lexicon.update(self.load_hf_seed_lexicon(dataset_name=dataset_name))
            except Exception:
                # Keep the pipeline runnable even if the external seed dataset fails.
                pass
        return lexicon

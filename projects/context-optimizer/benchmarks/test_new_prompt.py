"""Quick test: compare token efficiency — prose vs semantic core format."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_optimizer.compressor import BLOCK_SUMMARY_PROMPT, _estimate_tokens

SAMPLE = """
It is a truth universally acknowledged, that a single man in possession of a good
fortune, must be in want of a wife. Mr. Bennet of Longbourn had five daughters:
Elizabeth, Jane, Lydia, Kitty, and Mary. Mrs. Bennet's purpose was to marry them off.
Mr. Bingley rented Netherfield Park and fell for Jane at the local ball. His proud
friend Mr. Darcy dismissed local society but grew attracted to the witty Elizabeth Bennet.
Wickham told Elizabeth false stories about Darcy having cheated him of an inheritance.
Lydia eloped with Wickham at Brighton. Darcy secretly paid Wickham's debts to secure
the marriage. Elizabeth realised Darcy's true character and his love for her.
Darcy proposed twice — first at Hunsford, then at Pemberley. Elizabeth accepted.
""".strip()

old_triple = "TOPIC:Pride_Prejudice;PERSON:Elizabeth_Bennet,Mr_Darcy;REL:Darcy->attracted->Elizabeth;PLACE:Netherfield;EVENT:Longbourn_ball;CAUSE:Wickham_lies->Elizabeth_rejection"
old_prose = "Jane Austen's Pride and Prejudice opens with the Bennet family of Longbourn seeking marriages for five daughters. Mr. Bingley falls for Jane at Netherfield ball. His proud friend Darcy grows attracted to witty Elizabeth Bennet. Wickham's false claims about Darcy deceive Elizabeth. Lydia elopes with Wickham. Darcy intervenes, pays debts, secures marriage. Elizabeth and Darcy reconcile after two proposals."
new_core = "Bennet Longbourn five daughters Elizabeth Jane Lydia Kitty Mary; Bingley Netherfield ball attracted Jane; Darcy proud dismissed society attracted Elizabeth; Wickham false claims Darcy inheritance deceived Elizabeth; Lydia elopement Wickham Brighton; Darcy paid debts secured marriage; Elizabeth recognised Darcy character love; Darcy proposed Hunsford Pemberley; reconciliation marriage"

print("FORMAT COMPARISON — same facts, different representations")
print("=" * 60)
for label, text in [
    ("Old triples", old_triple),
    ("Prose", old_prose),
    ("Semantic core (new)", new_core),
]:
    toks = _estimate_tokens(text)
    filler = sum(
        1
        for w in text.split()
        if w.lower()
        in {
            "the",
            "a",
            "an",
            "of",
            "in",
            "at",
            "by",
            "for",
            "with",
            "from",
            "to",
            "and",
            "but",
            "or",
            "is",
            "was",
            "were",
            "has",
            "had",
            "be",
            "been",
            "will",
            "his",
            "her",
            "their",
            "it",
            "its",
            "he",
            "she",
            "they",
            "this",
            "that",
        }
    )
    signal = len(text.split()) - filler
    print(f"\n  {label}")
    print(f"    {text[:100]}...")
    print(
        f"    tokens={toks}  signal_words={signal}  filler={filler}  signal_ratio={signal/max(1,len(text.split())):.0%}"
    )

print()
print("In a 300-token budget:")
print(
    f"  Old triples: {300 // _estimate_tokens(old_triple) if _estimate_tokens(old_triple) else 'N/A'}x  -> {_estimate_tokens(old_triple)} tok used, ~{300 - _estimate_tokens(old_triple)} remaining"
)
print(
    f"  Prose      : {_estimate_tokens(old_prose)} tokens used, ~{300 - _estimate_tokens(old_prose)} remaining"
)
print(
    f"  Core (new) : {_estimate_tokens(new_core)} tokens used, ~{300 - _estimate_tokens(new_core)} remaining  <- {300 // max(1, _estimate_tokens(new_core))}x more facts fit"
)

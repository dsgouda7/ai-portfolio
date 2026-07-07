"""Quick test: show what the new prose prompt produces on a Gutenberg excerpt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_optimizer.compressor import BLOCK_SUMMARY_PROMPT, _estimate_tokens

# Sample text from Pride and Prejudice
SAMPLE = """
It is a truth universally acknowledged, that a single man in possession of a good
fortune, must be in want of a wife. However little known the feelings or views of such
a man may be on his first entering a neighbourhood, this truth is so well fixed in the
minds of the surrounding families, that he is considered as the rightful property of
some one or other of their daughters.

Mr. Bennet of Longbourn had five daughters: Elizabeth, Jane, Lydia, Kitty, and Mary.
Mrs. Bennet's sole purpose in life was to marry them off. When a wealthy young man named
Mr. Bingley rented Netherfield Park nearby, she immediately set her sights on him as a
match for one of her daughters. Mr. Bennet took quiet amusement in his wife's scheming.

Mr. Bingley soon fell in love with Jane Bennet at a local ball. His friend Mr. Darcy,
proud and aristocratic, initially dismissed the local society as beneath him, but
found himself increasingly attracted to the witty and independent Elizabeth Bennet.
""".strip()

prompt = BLOCK_SUMMARY_PROMPT.format(text=SAMPLE)
print("=== PROMPT (last 300 chars) ===")
print(prompt[-300:])
print()
print("=== PROMPT TOKEN ESTIMATE ===")
print(f"  {_estimate_tokens(prompt)} tokens in prompt")
print()

# Show comparison: old triple format example vs new prose example
old_output = "TOPIC:Pride_Prejudice;PERSON:Elizabeth_Bennet,Mr_Darcy,Mr_Bingley;REL:Darcy->attracted->Elizabeth;PLACE:Netherfield;EVENT:Longbourn_ball"
new_output = "Jane Austen's Pride and Prejudice opens with the Bennet family of Longbourn: Mrs. Bennet seeks husbands for five daughters including Elizabeth and Jane. When wealthy Mr. Bingley rents Netherfield Park, he falls for Jane at a local ball. His proud friend Mr. Darcy initially dismisses the neighbourhood but grows attracted to the witty Elizabeth Bennet."

print("=== OLD triple format ===")
print(f"  '{old_output}'")
print(f"  Tokens: {_estimate_tokens(old_output)}")
print()
print("=== NEW prose format ===")
print(f"  '{new_output}'")
print(f"  Tokens: {_estimate_tokens(new_output)}")
print()
print("Old format label overhead: TOPIC: PERSON: REL: PLACE: EVENT: -> ; = ~20 structural tokens")
print("New format: same budget, every token is a searchable word")
print()
print("Query 'Elizabeth Bennet Darcy' matches prose directly;")
print("  old format had 'Elizabeth_Bennet' (with underscore) which may not match embeddings cleanly")

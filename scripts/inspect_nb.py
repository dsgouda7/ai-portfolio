import json

with open("c:/repos/ai-portfolio/learning/genai/rnns/MIT/TF_Part1_Intro.ipynb") as f:
    nb = json.load(f)
cells = nb["cells"]
print("Total cells on disk:", len(cells))
print()
for i, c in enumerate(cells):
    cid = c.get("id", "NO_ID")
    src = "".join(c["source"])[:80].replace("\n", " ")
    ctype = c["cell_type"]
    print(f"Cell {i+1:2d} ({ctype[:4]}) [{cid}]: {src}")

import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open("baseline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "markdown":
        print("".join(cell.get("source", [])))
        print("-" * 40)
    elif cell.get("cell_type") == "code":
        print("CODE:")
        print("".join(cell.get("source", [])))
        print("-" * 40)

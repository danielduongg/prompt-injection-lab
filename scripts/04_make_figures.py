"""Step 4: render the ASR-by-category comparison figure."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import plot_asr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
s = json.load(open(ROOT / "results" / "eval_summary.json"))
plot_asr(s["no_defense"]["asr_by_category"], s["with_defense"]["asr_by_category"],
         ROOT / "results" / "figures" / "asr_by_category.png")
print("wrote results/figures/asr_by_category.png")

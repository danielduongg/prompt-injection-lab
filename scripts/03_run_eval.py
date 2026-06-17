"""Step 3: run the robustness harness with and without the detector filter."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib  # noqa: E402
from src.detector import DetectorFilter  # noqa: E402
from src.harness import run_eval, summarize  # noqa: E402
from src.models import get_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

attacks = json.load(open(ROOT / "data" / "attack_suite.json"))
target = get_model("mock")                       # swap for "anthropic" / "openai" / "hf"
det = DetectorFilter(joblib.load(RES / "detector.joblib"), threshold=0.5)

df_no = run_eval(target, attacks, detector=None)
df_def = run_eval(target, attacks, detector=det)
sum_no, sum_def = summarize(df_no), summarize(df_def)

df_no.to_csv(RES / "eval_no_defense.csv", index=False)
df_def.to_csv(RES / "eval_with_defense.csv", index=False)
json.dump({"target_model": target.name, "detector_threshold": 0.5,
           "no_defense": sum_no, "with_defense": sum_def},
          open(RES / "eval_summary.json", "w"), indent=2)

print(f"Target: {target.name}   (attacks={sum_no['n_attacks']}, benign={sum_no['n_benign']})")
print(f"  No defense  -> ASR overall {sum_no['asr_overall_pct']:.1f}%  "
      f"(seen {sum_no['asr_seen_families_pct']:.1f}% / held-out {sum_no['asr_heldout_families_pct']:.1f}%)")
print(f"  With filter -> ASR overall {sum_def['asr_overall_pct']:.1f}%  "
      f"(seen {sum_def['asr_seen_families_pct']:.1f}% / held-out {sum_def['asr_heldout_families_pct']:.1f}%)")
print(f"               attacks blocked {sum_def['attack_block_rate_pct']:.1f}%  |  "
      f"benign false-block {sum_def['benign_block_rate_pct']:.1f}%")

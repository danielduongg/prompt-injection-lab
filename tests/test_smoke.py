"""Fast smoke test: the pipeline trains a useful detector and the filter cuts ASR."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
from src import data_gen  # noqa: E402
from src.data_gen import SEEN_FAMILIES  # noqa: E402
from src.detector import DetectorFilter, train_and_eval  # noqa: E402
from src.harness import run_eval, summarize  # noqa: E402
from src.models import get_model  # noqa: E402


def test_pipeline_small():
    rng = data_gen._rng()
    rows = [(t, 0) for t in data_gen.gen_benign(120, rng)]
    rows += [(t, 1) for t in data_gen.gen_injections(120, rng, SEEN_FAMILIES)]
    df = pd.DataFrame(rows, columns=["text", "label"])

    model, metrics, _, _ = train_and_eval(df, test_size=0.3, seed=0)
    assert metrics["roc_auc"] > 0.85, metrics

    attacks = data_gen.build_attack_suite(per_cat=4, n_benign=30)
    target = get_model("mock")
    asr_no = summarize(run_eval(target, attacks))["asr_overall_pct"]
    asr_def = summarize(run_eval(target, attacks, detector=DetectorFilter(model)))["asr_overall_pct"]
    assert asr_def <= asr_no, (asr_def, asr_no)


if __name__ == "__main__":
    test_pipeline_small()
    print("smoke test passed")

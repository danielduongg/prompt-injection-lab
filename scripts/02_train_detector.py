"""Step 2: train the detector on SEEN families; evaluate in-distribution AND on the
held-out, adversarially-perturbed OOD test set. Save model, metrics, and figures."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from src.detector import evaluate, train_and_eval  # noqa: E402
from src.plots import plot_confusion, plot_roc_overlay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(ROOT / "data" / "injection_dataset.csv")
model, m_in, roc_in, _ = train_and_eval(train_df)

ood_df = pd.read_csv(ROOT / "data" / "ood_testset.csv")
m_ood, roc_ood, _ = evaluate(model, ood_df["text"].values, ood_df["label"].values)

joblib.dump(model, RES / "detector.joblib")
json.dump({"in_distribution": m_in, "ood_shift": m_ood},
          open(RES / "detector_metrics.json", "w"), indent=2)

plot_roc_overlay([("In-distribution", roc_in, m_in["roc_auc"]),
                  ("Distribution shift (held-out + perturbed)", roc_ood, m_ood["roc_auc"])],
                 RES / "figures" / "detector_roc.png")
plot_confusion(m_ood["confusion"], RES / "figures" / "detector_confusion_ood.png")
plot_confusion(m_in["confusion"], RES / "figures" / "detector_confusion_indist.png")


def row(tag, m):
    return (f"  {tag:18s} n={m['n']:<4d} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}")


print(f"Trained on {m_in['n_train']} prompts (seen families).")
print(row("in-distribution", m_in))
print(row("OOD shift", m_ood))

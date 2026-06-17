"""plots.py -- figures for the report (matplotlib, Agg backend, no display needed)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_roc(roc, metrics, path):
    plt.figure(figsize=(5, 4))
    plt.plot(roc["fpr"], roc["tpr"], lw=2, label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Prompt-injection detector -- ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(cm, path):
    cm = np.array(cm)
    plt.figure(figsize=(4.2, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    plt.xticks([0, 1], ["benign", "injection"])
    plt.yticks([0, 1], ["benign", "injection"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Detector confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_asr(no_def, with_def, path):
    cats = sorted(set(no_def) | set(with_def))
    x = np.arange(len(cats))
    w = 0.38
    plt.figure(figsize=(8.5, 4.6))
    plt.bar(x - w / 2, [no_def.get(c, 0) for c in cats], w,
            label="No defense", color="#c0392b")
    plt.bar(x + w / 2, [with_def.get(c, 0) for c in cats], w,
            label="With detector filter", color="#2980b9")
    plt.xticks(x, cats, rotation=30, ha="right")
    plt.ylabel("Attack success rate (%)")
    plt.title("Attack success rate by category")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_overlay(curves, path):
    """curves: list of (label, roc_dict, auc) tuples drawn on one axis."""
    plt.figure(figsize=(5.2, 4.2))
    for label, roc, auc in curves:
        plt.plot(roc["fpr"], roc["tpr"], lw=2, label=f"{label} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Detector ROC: in-distribution vs. distribution shift")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

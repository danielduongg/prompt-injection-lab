# Prompt-Injection Lab

[![CI](https://github.com/danielduongg/prompt-injection-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/danielduongg/prompt-injection-lab/actions)

**A lightweight prompt-injection detector + an LLM jailbreak-robustness evaluation harness — and an honest look at where cheap defenses fail.**

This repo is a small, fully reproducible testbed for AI-security research on
prompt injection. It (1) trains a transparent input-filter classifier, (2) runs a
categorized attack suite against a guarded target model to measure **Attack Success
Rate (ASR)**, and (3) stress-tests the filter under **distribution shift** to show how
misleading in-distribution metrics can be. It runs end-to-end **offline with no API
keys**, and swaps to real models (Anthropic / OpenAI / local HuggingFace) by changing
one line.

> **Headline finding.** The detector is *perfect* in-distribution (ROC-AUC = 1.000) but
> its recall **collapses to 0.22** under held-out, obfuscated attacks. As a filter it
> drives known-family ASR from **50.0% → 0.0%**, yet only **37.0% → 27.8%** on novel
> attacks — at a **3.6%** benign false-block cost. *In-distribution numbers dramatically
> overstate real robustness.* Full write-up: [`report/REPORT.md`](report/REPORT.md).

## Why this exists

Classifier-based input filters (e.g. Anthropic's *Constitutional Classifiers*) are a
front-line defense against prompt injection and jailbreaks. They are easy to make look
great on paper and easy to fool in practice. This project quantifies that gap with a
clean, judge-free methodology built around a benign **canary**: an attack "succeeds"
only if the guarded model leaks a meaningless secret token or goes off-task — so every
prompt here is harmless and safe to publish.

## Results at a glance

**Detector — in-distribution vs. distribution shift**

| Evaluation | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| In-distribution (seen families) | 1.000 | 1.000 | 1.000 | 1.000 |
| Shift (held-out families + adversarial perturbations) | 0.697 | 0.221 | 0.335 | 0.856 |

**Filter as a defense (144 attacks, 140 benign controls)**

| Condition | Overall ASR | Seen families | Held-out families | Benign false-block |
|---|---:|---:|---:|---:|
| No defense | 45.1% | 50.0% | 37.0% | — |
| + detector filter | 10.4% | 0.0% | 27.8% | 3.6% |

<p align="center">
  <img src="results/figures/asr_by_category.png" width="78%"><br>
  <img src="results/figures/detector_roc.png" width="48%">
</p>

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_all.py        # build data -> train detector -> evaluate -> figures
python tests/test_smoke.py       # fast end-to-end sanity check
```

Outputs land in `results/` (metrics JSON, per-item CSVs) and `results/figures/`.
Everything is seeded for reproducibility (`SEED = 20260617`).

### Evaluate a real model

```python
# scripts/03_run_eval.py
from src.models import get_model
target = get_model("anthropic", model="claude-3-5-sonnet-latest")  # needs ANTHROPIC_API_KEY
# target = get_model("openai", model="gpt-4o-mini")                # needs OPENAI_API_KEY
# target = get_model("hf", model="meta-llama/Llama-3.2-3B-Instruct")
```

## How it works

- **`src/data_gen.py`** — reproducibly generates a labeled detector dataset (5 *seen*
  attack families + benign), a held-out **OOD test set** (3 *unseen* families + novel
  payloads + homoglyph / zero-width-space / paraphrase perturbations), and a categorized
  `attack_suite.json` for the harness.
- **`src/detector.py`** — word+char TF-IDF → logistic-regression classifier and a
  `DetectorFilter` wrapper; transparent and trains in seconds.
- **`src/models.py`** — pluggable target interface. `MockModel` is a deterministic,
  offline guarded assistant that holds a canary and "reads through" obfuscation a lexical
  filter misses. Real backends: `AnthropicModel`, `OpenAIModel`, `HFModel`.
- **`src/harness.py`** — runs attacks (optionally behind the filter) and computes ASR
  overall / by category / seen-vs-held-out, plus benign false-block rate.
- **`src/refusal.py`** — exact, judge-free success detection via the canary.

```
prompt-injection-lab/
├── README.md
├── report/REPORT.md            # the research write-up (public output)
├── requirements.txt
├── src/                        # data_gen, detector, models, harness, refusal, plots
├── scripts/                    # 01_build_data → 02_train_detector → 03_run_eval → 04_make_figures (+ run_all)
├── data/                       # generated datasets + attack suite
├── results/                    # metrics, CSVs, figures/
└── tests/test_smoke.py
```

## Responsible use

All attacks target a **harmless canary / off-task** goal; nothing here seeks dangerous or
disallowed content. The repo is for **defensive** prompt-injection research.

## References

- Anthropic — *Constitutional Classifiers* (arXiv:2501.18837): https://arxiv.org/abs/2501.18837
- deepset — *prompt-injections* dataset: https://huggingface.co/datasets/deepset/prompt-injections
- *JailbreakBench* (NeurIPS 2024): https://jailbreakbench.github.io/
- NVIDIA — *garak* LLM vulnerability scanner: https://github.com/NVIDIA/garak

## License

MIT — see [`LICENSE`](LICENSE).

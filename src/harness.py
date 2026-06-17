"""
harness.py -- prompt-injection / jailbreak robustness evaluation harness.

Runs a suite of categorized attacks against a target model, optionally behind a
detector input-filter, and reports Attack Success Rate (ASR) overall, by category,
and split by whether the attack family was seen by the detector during training.
Also reports the benign block rate (false-positive rate of the filter).
"""
from __future__ import annotations

import pandas as pd

from .models import SYSTEM_INSTRUCTION
from .refusal import attack_succeeded


def run_eval(model, attacks, detector=None, system: str = SYSTEM_INSTRUCTION):
    rows = []
    for a in attacks:
        prompt, cat = a["prompt"], a["category"]
        blocked = success = False
        resp = ""
        if detector is not None and detector.flag(prompt):
            blocked = True
        else:
            resp = model.generate(system, prompt)
            success = attack_succeeded(resp)
        rows.append(dict(id=a["id"], category=cat,
                         is_attack=(cat != "benign"),
                         held_out=bool(a.get("held_out", False)),
                         blocked=blocked, success=success, response=resp[:200]))
    return pd.DataFrame(rows)


def _asr(series) -> float:
    return round(float(series.mean() * 100), 1) if len(series) else 0.0


def summarize(df: pd.DataFrame) -> dict:
    atk = df[df.is_attack]
    ben = df[~df.is_attack]
    seen = atk[~atk.held_out]
    held = atk[atk.held_out]
    by_cat = (atk.groupby("category")["success"].mean().mul(100).round(1)
              if len(atk) else pd.Series(dtype=float))
    return dict(
        n_attacks=int(len(atk)), n_benign=int(len(ben)),
        asr_overall_pct=_asr(atk["success"]) if len(atk) else 0.0,
        asr_seen_families_pct=_asr(seen["success"]) if len(seen) else 0.0,
        asr_heldout_families_pct=_asr(held["success"]) if len(held) else 0.0,
        attack_block_rate_pct=_asr(atk["blocked"]) if len(atk) else 0.0,
        benign_block_rate_pct=_asr(ben["blocked"]) if len(ben) else 0.0,
        asr_by_category={k: float(v) for k, v in by_cat.to_dict().items()},
    )

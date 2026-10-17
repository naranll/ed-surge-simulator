#!/usr/bin/env python
"""Rewrite models/model_card.md from the saved results (no retraining). Safe to rerun.

    python scripts/rewrite_cards.py
"""
import json

import pandas as pd

from _common import MODELS

import train_early_warning as a2
import train_patient_flow as a1

m = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))
if "patient_flow" in m:
    lookup = json.loads((MODELS / "patient_flow" / "admission_lookup.json").read_text(encoding="utf-8"))
    a1.write_card(MODELS / "model_card.md", m["patient_flow"], lookup)
if "early_warning" in m and (MODELS / "stage_results.csv").exists():
    a2.write_card(MODELS / "model_card.md", pd.read_csv(MODELS / "stage_results.csv"), m["early_warning"], None)
print("model_card.md rewritten")

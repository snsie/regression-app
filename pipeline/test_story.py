"""Check that the claims written into the app's text still hold for the generated results.

Run after run_all.py:  python test_story.py
If a check fails after re-running the pipeline (new data, new package versions), update the
wording in web/src/sections/ before publishing.
"""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent.parent / "web" / "public" / "data"


def j(name):
    return json.loads((DATA / f"{name}.json").read_text())


def rep(ds, split, model):
    return float(np.mean(ds["cv"][split]["models"][model]["repeated_r2"]))


def main():
    lab, glu, sbp, fev, pk = j("lab"), j("nhanes_glucose"), j("nhanes_sbp"), j("nhanes_fev1"), j("parkinsons")
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))

    h = lab["hook"]["verified"]["models"]
    check("Hook: forest training R² looks excellent (>0.8)", lab["hook"]["assistant"]["train_r2"] > 0.8)
    check("Hook: forest held-out R² far below training", h["rf"]["train_r2"] - h["rf"]["cv_r2"] > 0.4)
    check("Hook: linear held-out R² >= forest", h["linear"]["cv_r2"] >= h["rf"]["cv_r2"])

    runs = {(r["keep_total"], r["keep_id"], r["split"]): r for r in lab["leakage"]["runs"]}
    check("Leakage: assistant's code reports near-perfect training R²", runs[(True, True, "random")]["train_r2"] > 0.99)
    check("Leakage: fully fixed model is worse than the mean", runs[(False, False, "group")]["test_r2"] < 0)
    check("Leakage: held-out score with leaks still high", runs[(True, True, "random")]["test_r2"] > 0.95)

    rf = {s: pk["cv"][s]["models"]["rf"]["mean_test_r2"] for s in ("random", "temporal", "group")}
    check("Splits: random > temporal > group for the forest", rf["random"] > rf["temporal"] > rf["group"])
    best_group = max(rep(pk, "group", m) for m in ("linear", "rf", "gbm", "tabicl"))
    check("Splits/compare: no model beats the mean for new patients (best < 0.05)", best_group < 0.05)

    best_glu = max(rep(glu, "kfold", m) for m in ("linear", "rf", "gbm", "tabicl"))
    check("Compare: glucose, linear within 0.03 of the best", best_glu - rep(glu, "kfold", "linear") <= 0.03)
    check("Compare: lung function, every flexible model beats linear by > 0.3",
          min(rep(fev, "kfold", m) for m in ("rf", "gbm", "tabicl")) - rep(fev, "kfold", "linear") > 0.3)
    pg = j("playground")["pairs"]["age_fev1"]
    check("Framing/ladder: Age → FEV₁ line is nearly flat (R² < 0.05) but a forest bends (CV R² > 0.3)",
          pg["linear"]["train_r2"] < 0.05 and pg["rf"]["300_20"]["cv_r2"] > 0.3)
    flex = max(rep(pk, "temporal", m) for m in ("rf", "gbm", "tabicl"))
    check("Compare: known Parkinson's patients, flexible models beat linear by > 0.2", flex - rep(pk, "temporal", "linear") > 0.2)

    b = sbp["boosting"]
    i100 = b["rounds"].index(100)
    c = b["learning_rates"]["0.1"]
    check("Boosting: default LightGBM overfits blood pressure (gap > 0.3)", c["train_r2"][i100] - c["test_r2"][i100] > 0.3)

    n = lab["nested"]
    check("Nested: best_score_ is optimistic vs nested CV", n["best_score"] >= n["nested_mean"])

    width = max(len(k) for k, _ in checks)
    for k, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {k:<{width}}")
    failed = [k for k, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

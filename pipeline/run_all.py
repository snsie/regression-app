"""Train every model, run every verification analysis, and write JSON for the web app.

Usage:  python run_all.py            (full run, needs a CUDA GPU for TabICL)
        python run_all.py --quick    (fewer repeats, for development)

Output goes to ../web/public/data/.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.tree import DecisionTreeRegressor

import datasets as D
from models import MODEL_DETAILS, MODELS, N_FOLDS, SEED, SPLIT_SHORT, SPLITS, folds, make_model

OUT = Path(__file__).parent.parent / "web" / "public" / "data"
QUANTILES = [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]


# ---------------------------------------------------------------- helpers
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def rnd(a, d=2):
    """Round to a JSON-friendly list; NaN becomes null."""
    if np.isscalar(a):
        return None if a is None or not np.isfinite(a) else round(float(a), d)
    arr = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else round(float(v), d) for v in arr.ravel()] if arr.ndim == 1 else [rnd(r, d) for r in arr]


def fit_predict(name, Xtr, ytr, Xte, seed=SEED, quantiles=False):
    m = make_model(name, seed)
    m.fit(Xtr, ytr)
    out = {"train": m.predict(Xtr)}
    if name == "tabicl" and quantiles:
        p = m.predict(Xte, output_type=["mean", "quantiles"], alphas=QUANTILES)
        out["test"], out["q"] = p["mean"], p["quantiles"]
    else:
        out["test"] = m.predict(Xte)
    return out


def score(y_tr, p_tr, y_te, p_te):
    return {
        "train_r2": r2_score(y_tr, p_tr),
        "test_r2": r2_score(y_te, p_te),
        "test_rmse": rmse(y_te, p_te),
        "test_mae": mean_absolute_error(y_te, p_te),
    }


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- per-dataset analyses
def cross_validation(ds: D.Dataset, repeats: int):
    X, y = ds.X.to_numpy(float), ds.y.to_numpy(float)
    groups = None if ds.groups is None else ds.groups.to_numpy()
    tm = None if ds.time is None else ds.time.to_numpy()
    out = {}
    for split in ds.splits:
        log(f"  CV {ds.id} / {split}")
        n_rep = 1 if split == "temporal" else repeats
        res = {m: {"folds": [], "repeated_r2": [], "repeated_rmse": []} for m in MODELS}
        oof = {m: np.full(len(y), np.nan) for m in MODELS}
        fold_id = np.full(len(y), -1)
        lin_sigma = np.full(len(y), np.nan)
        tab_q = np.full((len(y), len(QUANTILES)), np.nan)
        for rep in range(n_rep):
            for k, (tr, te) in enumerate(folds(split, len(y), groups, tm, seed=SEED + rep)):
                if rep == 0:
                    fold_id[te] = k
                for m in MODELS:
                    p = fit_predict(m, X[tr], y[tr], X[te], seed=SEED, quantiles=(rep == 0))
                    s = score(y[tr], p["train"], y[te], p["test"])
                    res[m]["repeated_r2"].append(s["test_r2"])
                    res[m]["repeated_rmse"].append(s["test_rmse"])
                    if rep == 0:
                        res[m]["folds"].append({k2: rnd(v, 4) for k2, v in s.items()} | {"n_test": int(len(te))})
                        oof[m][te] = p["test"]
                        if m == "linear":
                            lin_sigma[te] = np.std(y[tr] - p["train"], ddof=X.shape[1] + 1)
                        if m == "tabicl" and "q" in p:
                            tab_q[te] = p["q"]
        models = {}
        for m in MODELS:
            f = res[m]["folds"]
            tr2 = [x["train_r2"] for x in f]
            te2 = [x["test_r2"] for x in f]
            models[m] = {
                "folds": f,
                "mean_train_r2": rnd(np.mean(tr2), 4),
                "mean_test_r2": rnd(np.mean(te2), 4),
                "sd_test_r2": rnd(np.std(te2, ddof=1), 4),
                "mean_test_rmse": rnd(np.mean([x["test_rmse"] for x in f]), 4),
                "sd_test_rmse": rnd(np.std([x["test_rmse"] for x in f], ddof=1), 4),
                "mean_test_mae": rnd(np.mean([x["test_mae"] for x in f]), 4),
                "repeated_r2": rnd(res[m]["repeated_r2"], 4),
                "repeated_rmse": rnd(res[m]["repeated_rmse"], 3),
            }
        dec = 3
        out[split] = {
            "label": SPLIT_SHORT[split],
            "description": SPLITS[split],
            "n_repeats": n_rep,
            "models": models,
            "oof": {
                "fold": fold_id.tolist(),
                "pred": {m: rnd(oof[m], dec) for m in MODELS},
                "linear_sigma": rnd(lin_sigma, dec),
                "tabicl_quantiles": rnd(tab_q, dec),
            },
        }
    return out


def learning_curves(ds: D.Dataset):
    log(f"  learning curves {ds.id}")
    X, y = ds.X.to_numpy(float), ds.y.to_numpy(float)
    groups = None if ds.groups is None else ds.groups.to_numpy()
    tm = None if ds.time is None else ds.time.to_numpy()
    fracs = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(SEED)
    out = {"fractions": fracs, "sizes": [], "models": {}}
    splits = folds(ds.main_split, len(y), groups, tm)
    out["sizes"] = [int(round(f * np.mean([len(tr) for tr, _ in splits]))) for f in fracs]
    for m in ["linear", "rf", "gbm", "tabicl"]:
        tr_scores, te_scores = [], []
        for f in fracs:
            a, b = [], []
            for tr, te in splits:
                sub = rng.choice(tr, size=max(20, int(round(f * len(tr)))), replace=False)
                p = fit_predict(m, X[sub], y[sub], X[te])
                a.append(r2_score(y[sub], p["train"]))
                b.append(r2_score(y[te], p["test"]))
            tr_scores.append(np.mean(a))
            te_scores.append(np.mean(b))
        out["models"][m] = {"train_r2": rnd(tr_scores, 4), "test_r2": rnd(te_scores, 4)}
    return out


BOOST_ROUNDS = list(range(1, 21)) + list(range(25, 101, 5)) + list(range(120, 501, 20))
BOOST_LRS = [0.01, 0.05, 0.1, 0.3]


def boosting_staged(ds: D.Dataset):
    log(f"  boosting staged {ds.id}")
    X, y = ds.X.to_numpy(float), ds.y.to_numpy(float)
    groups = None if ds.groups is None else ds.groups.to_numpy()
    tm = None if ds.time is None else ds.time.to_numpy()
    splits = folds(ds.main_split, len(y), groups, tm)
    out = {"rounds": BOOST_ROUNDS, "learning_rates": {}}
    for lr in BOOST_LRS:
        tr_all, te_all = [], []
        for tr, te in splits:
            m = LGBMRegressor(n_estimators=max(BOOST_ROUNDS), learning_rate=lr, random_state=SEED, verbose=-1, n_jobs=8)
            m.fit(X[tr], y[tr])
            tr_all.append([r2_score(y[tr], m.predict(X[tr], num_iteration=r)) for r in BOOST_ROUNDS])
            te_all.append([r2_score(y[te], m.predict(X[te], num_iteration=r)) for r in BOOST_ROUNDS])
        out["learning_rates"][str(lr)] = {
            "train_r2": rnd(np.mean(tr_all, axis=0), 4),
            "test_r2": rnd(np.mean(te_all, axis=0), 4),
        }
    return out


RF_TREES = [1, 2, 3, 5, 10, 20, 50, 100, 200, 300]
RF_LEAVES = [1, 5, 20]


def forest_size(ds: D.Dataset):
    log(f"  forest size {ds.id}")
    X, y = ds.X.to_numpy(float), ds.y.to_numpy(float)
    groups = None if ds.groups is None else ds.groups.to_numpy()
    tm = None if ds.time is None else ds.time.to_numpy()
    splits = folds(ds.main_split, len(y), groups, tm)
    out = {"n_trees": RF_TREES, "min_samples_leaf": {}}
    for leaf in RF_LEAVES:
        tr_all, te_all = [], []
        for tr, te in splits:
            m = RandomForestRegressor(n_estimators=max(RF_TREES), min_samples_leaf=leaf, n_jobs=-1, random_state=SEED)
            m.fit(X[tr], y[tr])
            ptr = np.cumsum([t.predict(X[tr]) for t in m.estimators_], axis=0)
            pte = np.cumsum([t.predict(X[te]) for t in m.estimators_], axis=0)
            tr_all.append([r2_score(y[tr], ptr[n - 1] / n) for n in RF_TREES])
            te_all.append([r2_score(y[te], pte[n - 1] / n) for n in RF_TREES])
        out["min_samples_leaf"][str(leaf)] = {
            "train_r2": rnd(np.mean(tr_all, axis=0), 4),
            "test_r2": rnd(np.mean(te_all, axis=0), 4),
        }
    return out


def linear_recipe(ds: D.Dataset):
    """Unscaled linear fit on all rows: the 'recipe of fixed weights'."""
    lr = LinearRegression().fit(ds.X, ds.y)
    feats = []
    for c, w in zip(ds.X.columns, lr.coef_):
        col = ds.X[c]
        binary = set(np.unique(col)) <= {0.0, 1.0}
        feats.append({
            "id": c,
            "label": ds.feature_labels.get(c, c),
            "weight": rnd(w, 5),
            "binary": bool(binary),
            "min": rnd(col.quantile(0.01), 3),
            "max": rnd(col.quantile(0.99), 3),
            "median": rnd(col.median(), 3),
            "mean": rnd(col.mean(), 4),
        })
    return {"intercept": rnd(lr.intercept_, 4), "features": feats, "train_r2": rnd(lr.score(ds.X, ds.y), 4)}


def dataset_json(ds: D.Dataset, repeats: int):
    log(f"dataset {ds.id}: n={len(ds.y)}")
    y = ds.y.to_numpy(float)
    return {
        "id": ds.id,
        "name": ds.name,
        "story": ds.story,
        "source": ds.source,
        "citation": ds.citation,
        "url": ds.url,
        "target": ds.target,
        "target_label": ds.target_label,
        "units": ds.units,
        "n_rows": int(len(y)),
        "n_patients": int(ds.groups.nunique()) if ds.groups is not None else int(len(y)),
        "features": [{"id": c, "label": ds.feature_labels.get(c, c)} for c in ds.X.columns],
        "subgroup_name": ds.subgroup_name,
        "subgroup": ds.subgroup.astype(str).tolist(),
        "y": rnd(y, 3),
        "y_summary": {"mean": rnd(y.mean(), 3), "sd": rnd(y.std(ddof=1), 3), "min": rnd(y.min(), 3), "max": rnd(y.max(), 3)},
        "splits": ds.splits,
        "main_split": ds.main_split,
        "cv": cross_validation(ds, repeats),
        "learning_curve": learning_curves(ds),
        "boosting": boosting_staged(ds),
        "forest": forest_size(ds),
        "recipe": linear_recipe(ds),
    }


# ---------------------------------------------------------------- playground (1 predictor)
PG_RF = [(n, leaf) for n in [1, 3, 10, 30, 100, 300] for leaf in [1, 5, 20]]
PG_GB_ROUNDS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
PG_GB_LRS = [0.03, 0.1, 0.3]


def _cv_r2_1d(make, x, y):
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    r = cross_validate(make(), x[:, None], y, cv=cv, scoring="r2", return_train_score=True)
    return rnd(r["train_score"].mean(), 4), rnd(r["test_score"].mean(), 4)


def playground():
    log("playground")
    out = {}
    for key, p in D.playground_pairs().items():
        x, y = p["x"].astype(float), p["y"].astype(float)
        lo, hi = x.min(), x.max()
        span = hi - lo
        grid = np.linspace(max(0.0, lo - 0.35 * span), hi + 0.35 * span, 140)  # every predictor is positive
        G = grid[:, None]
        entry = {k: v for k, v in p.items() if k not in ("x", "y")}
        entry |= {"x": rnd(x, 2), "y": rnd(y, 1), "grid": rnd(grid, 3), "data_range": [rnd(lo, 2), rnd(hi, 2)]}

        lin = LinearRegression().fit(x[:, None], y)
        entry["linear"] = {"intercept": rnd(lin.intercept_, 4), "slope": rnd(lin.coef_[0], 4)}
        entry["linear"]["train_r2"], entry["linear"]["cv_r2"] = _cv_r2_1d(LinearRegression, x, y)

        rf = {}
        for n, leaf in PG_RF:
            mk = lambda n=n, leaf=leaf: RandomForestRegressor(n_estimators=n, min_samples_leaf=leaf, random_state=SEED, n_jobs=-1)
            m = mk().fit(x[:, None], y)
            tr, cv = _cv_r2_1d(mk, x, y)
            rf[f"{n}_{leaf}"] = {"curve": rnd(m.predict(G), 2), "train_r2": tr, "cv_r2": cv}
        entry["rf"] = rf

        gb = {}
        for lr in PG_GB_LRS:
            m = LGBMRegressor(n_estimators=max(PG_GB_ROUNDS), learning_rate=lr, num_leaves=4, min_child_samples=5,
                              random_state=SEED, verbose=-1, n_jobs=4).fit(x[:, None], y)
            cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
            fold_models = [(LGBMRegressor(n_estimators=max(PG_GB_ROUNDS), learning_rate=lr, num_leaves=4, min_child_samples=5,
                                          random_state=SEED, verbose=-1, n_jobs=4).fit(x[tr, None], y[tr]), tr, te)
                           for tr, te in cv.split(x)]
            for r in PG_GB_ROUNDS:
                if r == 0:
                    curve = np.full_like(grid, y.mean())
                    tr_r2 = 0.0
                    cv_r2 = np.mean([r2_score(y[te], np.full(len(te), y[tr].mean())) for _, tr, te in fold_models])
                else:
                    curve = m.predict(G, num_iteration=r)
                    tr_r2 = np.mean([r2_score(y[tr], fm.predict(x[tr, None], num_iteration=r)) for fm, tr, te in fold_models])
                    cv_r2 = np.mean([r2_score(y[te], fm.predict(x[te, None], num_iteration=r)) for fm, tr, te in fold_models])
                gb[f"{lr}_{r}"] = {"curve": rnd(curve, 2), "train_r2": rnd(tr_r2, 4), "cv_r2": rnd(cv_r2, 4)}
        entry["gbm"] = gb

        from tabicl import TabICLRegressor

        tab = TabICLRegressor(random_state=SEED, device="cuda").fit(x[:, None], y)
        pq = tab.predict(G, output_type=["mean", "quantiles"], alphas=[0.05, 0.95])
        cvs = []
        for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(x):
            t = TabICLRegressor(random_state=SEED, device="cuda").fit(x[tr, None], y[tr])
            cvs.append(r2_score(y[te], t.predict(x[te, None])))
        entry["tabicl"] = {
            "curve": rnd(pq["mean"], 2),
            "lo": rnd(pq["quantiles"][:, 0], 2),
            "hi": rnd(pq["quantiles"][:, 1], 2),
            "train_r2": rnd(r2_score(y, tab.predict(x[:, None])), 4),
            "cv_r2": rnd(np.mean(cvs), 4),
        }
        out[key] = entry
    return {"pairs": out, "rf_trees": [1, 3, 10, 30, 100, 300], "rf_leaves": [1, 5, 20],
            "gb_rounds": PG_GB_ROUNDS, "gb_lrs": PG_GB_LRS}


def export_tree(t: DecisionTreeRegressor, names):
    tr = t.tree_

    def node(i):
        if tr.children_left[i] == -1:
            return {"leaf": True, "value": rnd(tr.value[i][0][0], 1), "n": int(tr.n_node_samples[i])}
        return {
            "leaf": False,
            "feature": names[tr.feature[i]],
            "threshold": rnd(tr.threshold[i], 1),
            "n": int(tr.n_node_samples[i]),
            "left": node(tr.children_left[i]),
            "right": node(tr.children_right[i]),
        }

    return node(0)


def tree_walkthrough():
    log("tree walk-through")
    d = D.nhanes_two_feature()
    X, y = d[["RIDAGEYR", "BMXBMI"]].to_numpy(float), d.SBP.to_numpy(float)
    names = ["age", "bmi"]
    small = RandomForestRegressor(n_estimators=3, max_depth=2, max_features=1, random_state=3).fit(X, y)
    big = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=SEED, n_jobs=-1).fit(X, y)
    ages = np.arange(18, 81, 2)
    bmis = np.arange(16, 56, 2)
    grid = np.array([[a, b] for a in ages for b in bmis], dtype=float)
    return {
        "n": int(len(y)),
        "mean_sbp": rnd(y.mean(), 1),
        "trees": [export_tree(t, names) for t in small.estimators_],
        "forest_grid": {"ages": ages.tolist(), "bmis": bmis.tolist(),
                        "pred": rnd(big.predict(grid).reshape(len(ages), len(bmis)), 1)},
        "points": {"age": rnd(X[:, 0], 0), "bmi": rnd(X[:, 1], 1), "sbp": rnd(y, 0)},
    }


# ---------------------------------------------------------------- lab: hook, leakage, nested CV
def hook():
    """What an unsupervised 'assistant' would report, versus a verified answer (NHANES SBP)."""
    log("hook")
    ds = D.nhanes_sbp()
    X, y = ds.X, ds.y.to_numpy(float)
    patient = {c: 0.0 for c in X.columns}
    patient.update({"RIDAGEYR": 55, "BMXBMI": 28, "BMXWAIST": 98, "female": 1, "INDFMPIR": 2.5})
    P = pd.DataFrame([patient])[X.columns]
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X, y)
    gbm = LGBMRegressor(random_state=SEED, verbose=-1).fit(X, y)
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    r = {}
    for name, mk in [("rf", lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
                     ("gbm", lambda: LGBMRegressor(random_state=SEED, verbose=-1)),
                     ("linear", lambda: make_model("linear")),
                     ("baseline", lambda: make_model("baseline"))]:
        s = cross_validate(mk(), X, y, cv=cv, scoring=("r2", "neg_root_mean_squared_error"), return_train_score=True)
        r[name] = {"train_r2": rnd(s["train_r2"].mean(), 3), "cv_r2": rnd(s["test_r2"].mean(), 3),
                   "cv_rmse": rnd(-s["test_neg_root_mean_squared_error"].mean(), 2)}
    lin = make_model("linear").fit(X, y)
    return {
        "patient": {"age": 55, "bmi": 28, "waist": 98, "sex": "Female", "income_ratio": 2.5},
        "assistant": {"model": "Random forest", "prediction": rnd(rf.predict(P)[0], 0), "train_r2": rnd(rf.score(X, y), 3),
                      "gbm_train_r2": rnd(gbm.score(X, y), 3)},
        "verified": {"linear_prediction": rnd(lin.predict(P)[0], 0), "models": r},
        "n": int(len(y)),
        "bmi_only_r2": None,
    }


def leakage_lab():
    log("leakage lab")
    d = D.parkinsons_raw()
    y = d.motor_UPDRS.to_numpy(float)
    groups = d["subject#"].to_numpy()
    base_feats = [c for c in d.columns if c not in ("subject#", "motor_UPDRS", "total_UPDRS")]
    out = []
    for keep_total in (True, False):
        for keep_id in (True, False):
            feats = base_feats + (["total_UPDRS"] if keep_total else []) + (["subject#"] if keep_id else [])
            X = d[feats].to_numpy(float)
            for split in ("random", "group"):
                tr2, te2, base = [], [], []
                for tr, te in folds(split, len(y), groups):
                    m = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED).fit(X[tr], y[tr])
                    tr2.append(m.score(X[tr], y[tr]))
                    te2.append(m.score(X[te], y[te]))
                    base.append(r2_score(y[te], np.full(len(te), y[tr].mean())))
                out.append({"keep_total": keep_total, "keep_id": keep_id, "split": split,
                            "train_r2": rnd(np.mean(tr2), 4), "test_r2": rnd(np.mean(te2), 4),
                            "test_r2_folds": rnd(te2, 4), "baseline_r2": rnd(np.mean(base), 4)})
                log(f"    total={keep_total} id={keep_id} {split}: train {np.mean(tr2):.3f} test {np.mean(te2):.3f}")
    corr = float(np.corrcoef(d.motor_UPDRS, d.total_UPDRS)[0, 1])
    return {"runs": out, "corr_motor_total": rnd(corr, 4), "n_rows": int(len(y)), "n_patients": int(len(np.unique(groups)))}


def nested_cv():
    log("nested CV (diabetes, LightGBM)")
    ds = D.diabetes()
    X, y = ds.X.to_numpy(float), ds.y.to_numpy(float)
    grid = {"num_leaves": [4, 8, 16, 31], "learning_rate": [0.01, 0.05, 0.1], "n_estimators": [100, 300]}
    outer = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    inner = KFold(N_FOLDS, shuffle=True, random_state=SEED + 1)
    base = LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=4)
    flat = GridSearchCV(base, grid, cv=outer, scoring="r2", n_jobs=4).fit(X, y)
    outer_scores, picks = [], []
    for tr, te in outer.split(X):
        gs = GridSearchCV(base, grid, cv=inner, scoring="r2", n_jobs=4).fit(X[tr], y[tr])
        outer_scores.append(gs.score(X[te], y[te]))
        picks.append({k: (float(v) if isinstance(v, float) else int(v)) for k, v in gs.best_params_.items()})
    untuned = cross_validate(base, X, y, cv=outer, scoring="r2")["test_score"]
    lin = cross_validate(make_model("linear"), X, y, cv=outer, scoring="r2")["test_score"]
    n_settings = int(np.prod([len(v) for v in grid.values()]))
    return {
        "grid": grid,
        "n_settings": n_settings,
        "best_score": rnd(flat.best_score_, 4),
        "best_params": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in flat.best_params_.items()},
        "nested_scores": rnd(outer_scores, 4),
        "nested_mean": rnd(np.mean(outer_scores), 4),
        "nested_sd": rnd(np.std(outer_scores, ddof=1), 4),
        "outer_picks": picks,
        "untuned_mean": rnd(untuned.mean(), 4),
        "linear_mean": rnd(lin.mean(), 4),
        "all_settings": [{"params": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in p.items()},
                          "score": rnd(s, 4)} for p, s in zip(flat.cv_results_["params"], flat.cv_results_["mean_test_score"])],
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 repeats instead of 10")
    ap.add_argument("--only", nargs="*", help="subset of outputs: datasets playground lab")
    args = ap.parse_args()
    repeats = 2 if args.quick else 10
    only = set(args.only or ["datasets", "playground", "lab"])
    t0 = time.time()
    D.download()
    OUT.mkdir(parents=True, exist_ok=True)

    def write(name, obj):
        (OUT / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
        log(f"wrote {name}.json ({(OUT / f'{name}.json').stat().st_size / 1e3:.0f} kB)")

    if "datasets" in only:
        for ds in D.all_datasets():
            write(ds.id, dataset_json(ds, repeats))
    if "playground" in only:
        write("playground", playground() | {"trees": tree_walkthrough()})
    if "lab" in only:
        h = hook()
        pg = D.playground_pairs()["bmi_sbp"]
        h["bmi_only_r2"] = rnd(np.corrcoef(pg["x"], pg["y"])[0, 1] ** 2, 3)
        write("lab", {"hook": h, "leakage": leakage_lab(), "nested": nested_cv()})

    pkgs = ["numpy", "pandas", "scikit-learn", "lightgbm", "tabicl", "torch"]
    write("meta", {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "repeats": repeats,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in pkgs},
        "models": MODELS,
        "model_details": MODEL_DETAILS,
        "splits": SPLITS,
        "datasets": [ds.id for ds in D.all_datasets()],
        "runtime_seconds": round(time.time() - t0),
    })


if __name__ == "__main__":
    main()

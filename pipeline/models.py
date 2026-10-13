"""The model ladder and the split strategies, defined once and shared by every analysis."""

from __future__ import annotations

import warnings

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

SEED = 0
N_FOLDS = 5

MODELS = {
    "baseline": "Baseline (mean)",
    "linear": "Linear regression",
    "rf": "Random forest",
    "gbm": "Gradient boosting",
    "tabicl": "Foundation model (TabICL)",
}

MODEL_DETAILS = {
    "baseline": "DummyRegressor(): predicts the training mean for everyone",
    "linear": "make_pipeline(StandardScaler(), LinearRegression())",
    "rf": "RandomForestRegressor(n_estimators=300, random_state=0)",
    "gbm": "LGBMRegressor(random_state=0) — defaults: 100 trees, learning rate 0.1, 31 leaves",
    "tabicl": "TabICLRegressor(random_state=0) — pretrained tabular foundation model, Apache-2.0",
}


def make_model(name: str, seed: int = SEED):
    if name == "baseline":
        return DummyRegressor()
    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "gbm":
        return LGBMRegressor(random_state=seed, verbose=-1, n_jobs=8)
    if name == "tabicl":
        from tabicl import TabICLRegressor

        return TabICLRegressor(random_state=seed, device="cuda", verbose=False)
    raise ValueError(name)


SPLITS = {
    "kfold": "KFold(5, shuffle=True): one row per patient",
    "random": "Random rows: KFold(5, shuffle=True), ignoring patient ID",
    "temporal": "Known patients, later visits: each patient's recordings in time order; train on earlier visits, test on the next block",
    "group": "New patients: GroupKFold(5) by patient ID",
}

SPLIT_SHORT = {
    "kfold": "K-fold",
    "random": "Random rows",
    "temporal": "Track known patients",
    "group": "Screen new patients",
}


def temporal_folds(groups: np.ndarray, time: np.ndarray, n_splits: int = N_FOLDS):
    """TimeSeriesSplit applied inside every patient, pooled across patients.

    Each patient's recordings are cut into n_splits + 1 time-ordered blocks.
    Fold k trains on blocks 0..k-1 of every patient and tests on block k.
    """
    block = np.zeros(len(groups), dtype=int)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        order = idx[np.argsort(time[idx], kind="stable")]
        block[order] = np.floor(np.arange(len(order)) * (n_splits + 1) / len(order)).astype(int)
    return [(np.where(block < k)[0], np.where(block == k)[0]) for k in range(1, n_splits + 1)]


def folds(split: str, n: int, groups=None, time=None, seed: int = SEED):
    """Return a list of (train_idx, test_idx) for one repetition of a split strategy."""
    idx = np.arange(n)
    if split in ("kfold", "random"):
        return list(KFold(N_FOLDS, shuffle=True, random_state=seed).split(idx))
    if split == "group":
        return list(GroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(idx, groups=groups))
    if split == "temporal":
        return temporal_folds(np.asarray(groups), np.asarray(time))
    raise ValueError(split)

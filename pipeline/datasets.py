"""Load the public datasets used by the app.

Every dataset comes from a public, citable source:
  * NHANES 2017-2018 (CDC National Center for Health Statistics), public-use files
  * NHANES 2011-2012 spirometry (the most recent cycle with lung function tests)
  * sklearn's Diabetes data (used only for the nested-CV demonstration) (Efron et al., Annals of Statistics 2004)
  * UCI Parkinson's Telemonitoring (Tsanas et al., IEEE TBME 2010)

Raw files are downloaded once into ``raw/`` by ``download()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import load_diabetes

RAW = Path(__file__).parent / "raw"

NHANES_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{}.xpt"
NHANES_FILES = ["DEMO_J", "BMX_J", "BPX_J", "GHB_J", "GLU_J"]
NHANES_2011_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/{}.xpt"
NHANES_2011_FILES = ["DEMO_G", "BMX_G", "SPX_G"]
PARKINSONS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/telemonitoring/parkinsons_updrs.data"
)


def download() -> None:
    RAW.mkdir(exist_ok=True)
    targets = {f"{f}.xpt": NHANES_URL.format(f) for f in NHANES_FILES}
    targets |= {f"{f}.xpt": NHANES_2011_URL.format(f) for f in NHANES_2011_FILES}
    targets["parkinsons_updrs.data"] = PARKINSONS_URL
    for name, url in targets.items():
        path = RAW / name
        if path.exists():
            continue
        print(f"downloading {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)


@dataclass
class Dataset:
    id: str
    name: str
    story: str
    source: str
    citation: str
    url: str
    target: str
    target_label: str
    units: str
    X: pd.DataFrame
    y: pd.Series
    feature_labels: dict[str, str]
    subgroup: pd.Series  # labelled subgroup per row, for subgroup checks
    subgroup_name: str
    groups: pd.Series | None = None  # patient id when rows repeat
    time: pd.Series | None = None  # within-patient time, for temporal splits
    splits: list[str] = field(default_factory=lambda: ["kfold"])
    main_split: str = "kfold"


def _nhanes_adults() -> pd.DataFrame:
    def read(f: str) -> pd.DataFrame:
        return pd.read_sas(RAW / f"{f}.xpt").set_index("SEQN")

    d = (
        read("DEMO_J")[["RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR"]]
        .join(read("BMX_J")[["BMXBMI", "BMXWAIST", "BMXHT", "BMXWT"]])
        .join(read("BPX_J")[["BPXSY1", "BPXSY2", "BPXSY3"]])
        .join(read("GHB_J")[["LBXGH"]])
        .join(read("GLU_J")[["LBXGLU"]], how="left")
    )
    d = d[d.RIDAGEYR >= 18].copy()
    d["SBP"] = d[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
    d["female"] = (d.RIAGENDR == 2).astype(float)
    d["sex_label"] = np.where(d.female == 1, "Female", "Male")
    return d


def _nhanes_spirometry() -> pd.DataFrame:
    """NHANES 2011-2012 participants aged 6-79 with a complete baseline spirometry test."""
    def read(f: str) -> pd.DataFrame:
        return pd.read_sas(RAW / f"{f}.xpt").set_index("SEQN")

    d = (
        read("SPX_G")[["SPXNSTAT", "SPXNFEV1"]]
        .join(read("DEMO_G")[["RIAGENDR", "RIDAGEYR", "INDFMPIR"]])
        .join(read("BMX_G")[["BMXBMI"]])
    )
    d = d[(d.SPXNSTAT == 1) & d.SPXNFEV1.notna()].copy()
    d["female"] = (d.RIAGENDR == 2).astype(float)
    return d


NHANES_CITE = (
    "Centers for Disease Control and Prevention, National Center for Health "
    "Statistics. National Health and Nutrition Examination Survey 2017-2018."
)
NHANES_LINK = "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017"

ETHNICITY = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other / multiracial",
}


def nhanes_glucose() -> Dataset:
    d = _nhanes_adults()
    cols = ["LBXGH", "RIDAGEYR", "BMXBMI", "BMXWAIST", "female"]
    s = d[cols + ["LBXGLU", "sex_label"]].dropna()
    return Dataset(
        id="nhanes_glucose",
        name="NHANES: HbA1c → fasting glucose",
        story="Nearly linear",
        source="NHANES 2017-2018 (CDC), adults in the fasting subsample",
        citation=NHANES_CITE,
        url=NHANES_LINK,
        target="LBXGLU",
        target_label="Fasting glucose",
        units="mg/dL",
        X=s[cols].reset_index(drop=True),
        y=s.LBXGLU.reset_index(drop=True),
        feature_labels={
            "LBXGH": "HbA1c (%)",
            "RIDAGEYR": "Age (years)",
            "BMXBMI": "BMI (kg/m²)",
            "BMXWAIST": "Waist (cm)",
            "female": "Female (1 = yes)",
        },
        subgroup=s.sex_label.reset_index(drop=True),
        subgroup_name="Sex",
    )


def nhanes_sbp() -> Dataset:
    d = _nhanes_adults()
    d = d[d.RIDRETH3.isin(ETHNICITY)]
    cols = ["RIDAGEYR", "BMXBMI", "BMXWAIST", "female", "INDFMPIR"]
    s = d[cols + ["SBP", "RIDRETH3"]].dropna().copy()
    eth = pd.get_dummies(s.RIDRETH3.map(ETHNICITY), prefix="eth", dtype=float)
    eth = eth.drop(columns="eth_Non-Hispanic White")
    X = pd.concat([s[cols], eth], axis=1)
    labels = {
        "RIDAGEYR": "Age (years)",
        "BMXBMI": "BMI (kg/m²)",
        "BMXWAIST": "Waist (cm)",
        "female": "Female (1 = yes)",
        "INDFMPIR": "Income-to-poverty ratio",
    }
    for c in eth.columns:
        labels[c] = c.replace("eth_", "") + " (vs NH White)"
    return Dataset(
        id="nhanes_sbp",
        name="NHANES: systolic blood pressure",
        story="Weak signal",
        source="NHANES 2017-2018 (CDC), adults with complete measurements",
        citation=NHANES_CITE,
        url=NHANES_LINK,
        target="SBP",
        target_label="Systolic blood pressure (mean of 3 readings)",
        units="mmHg",
        X=X.reset_index(drop=True),
        y=s.SBP.reset_index(drop=True),
        feature_labels=labels,
        subgroup=s.RIDRETH3.map(ETHNICITY).reset_index(drop=True),
        subgroup_name="Race / ethnicity",
    )


NHANES_2011_CITE = (
    "Centers for Disease Control and Prevention, National Center for Health "
    "Statistics. National Health and Nutrition Examination Survey 2011-2012."
)
NHANES_2011_LINK = "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011"


def nhanes_fev1() -> Dataset:
    d = _nhanes_spirometry()
    cols = ["RIDAGEYR", "female", "BMXBMI", "INDFMPIR"]
    s = d[cols + ["SPXNFEV1"]].dropna()
    band = pd.cut(s.RIDAGEYR, [0, 17, 39, 59, 120], labels=["Age 6–17", "Age 18–39", "Age 40–59", "Age 60–79"])
    return Dataset(
        id="nhanes_fev1",
        name="NHANES: lung function (FEV₁)",
        story="Curved",
        source="NHANES 2011-2012 (CDC), ages 6-79 with complete spirometry. Height is left out on purpose, so the age curve has to be learned",
        citation=NHANES_2011_CITE,
        url=NHANES_2011_LINK,
        target="SPXNFEV1",
        target_label="FEV₁ (air exhaled in the first second)",
        units="mL",
        X=s[cols].reset_index(drop=True),
        y=s.SPXNFEV1.reset_index(drop=True),
        feature_labels={
            "RIDAGEYR": "Age (years)",
            "female": "Female (1 = yes)",
            "BMXBMI": "BMI (kg/m²)",
            "INDFMPIR": "Income-to-poverty ratio",
        },
        subgroup=band.astype(str).reset_index(drop=True),
        subgroup_name="Age group",
    )


def diabetes() -> Dataset:
    raw = load_diabetes(as_frame=True, scaled=False)
    X = raw.data.rename(columns={"s1": "tc", "s2": "ldl", "s3": "hdl", "s4": "tch", "s5": "ltg", "s6": "glu"})
    sex = np.where(X["sex"] == 1, "Sex code 1", "Sex code 2")
    return Dataset(
        id="diabetes",
        name="Diabetes progression (442 patients)",
        story="Small n",
        source="scikit-learn Diabetes dataset (Efron et al. 2004)",
        citation="Efron B, Hastie T, Johnstone I, Tibshirani R. Least Angle Regression. Ann Stat 2004;32(2):407-99.",
        url="https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset",
        target="progression",
        target_label="Disease progression one year after baseline",
        units="points",
        X=X,
        y=raw.target.rename("progression"),
        feature_labels={
            "age": "Age (years)",
            "sex": "Sex (code)",
            "bmi": "BMI (kg/m²)",
            "bp": "Mean arterial pressure",
            "tc": "Total cholesterol",
            "ldl": "LDL",
            "hdl": "HDL",
            "tch": "Total / HDL cholesterol",
            "ltg": "log triglycerides",
            "glu": "Blood glucose",
        },
        subgroup=pd.Series(sex),
        subgroup_name="Sex (as coded)",
    )


def parkinsons() -> Dataset:
    d = pd.read_csv(RAW / "parkinsons_updrs.data")
    feats = [c for c in d.columns if c not in ("subject#", "motor_UPDRS", "total_UPDRS")]
    labels = {c: c for c in feats}
    labels.update({"age": "Age (years)", "sex": "Sex (code)", "test_time": "Days since recruitment"})
    return Dataset(
        id="parkinsons",
        name="Parkinson's telemonitoring (42 patients)",
        story="Repeated patients",
        source="UCI Parkinson's Telemonitoring (Tsanas et al. 2010)",
        citation=(
            "Tsanas A, Little MA, McSharry PE, Ramig LO. Accurate telemonitoring of "
            "Parkinson's disease progression by non-invasive speech tests. "
            "IEEE Trans Biomed Eng 2010;57(4):884-93."
        ),
        url="https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring",
        target="motor_UPDRS",
        target_label="Motor UPDRS",
        units="points",
        X=d[feats],
        y=d.motor_UPDRS,
        feature_labels=labels,
        subgroup=pd.Series(np.where(d.sex == 0, "Sex code 0", "Sex code 1")),
        subgroup_name="Sex (as coded)",
        groups=d["subject#"],
        time=d.test_time,
        splits=["random", "temporal", "group"],
        main_split="group",
    )


def parkinsons_raw() -> pd.DataFrame:
    return pd.read_csv(RAW / "parkinsons_updrs.data")


def playground_pairs(n: int = 300, seed: int = 0) -> dict[str, dict]:
    """Small one-predictor samples for the hands-on line and model-ladder views."""
    adults = _nhanes_adults()
    spiro = _nhanes_spirometry()
    males = spiro[spiro.female == 0]
    adults_desc = "NHANES 2017–2018 adults"
    pairs = {
        "age_sbp": (adults, adults_desc, "RIDAGEYR", "SBP", "Age", "years", "Systolic BP", "mmHg"),
        "bmi_sbp": (adults, adults_desc, "BMXBMI", "SBP", "BMI", "kg/m²", "Systolic BP", "mmHg"),
        "a1c_glucose": (adults, adults_desc, "LBXGH", "LBXGLU", "HbA1c", "%", "Fasting glucose", "mg/dL"),
        # one sex only, so the male/female gap in lung size doesn't blur the age curve
        "age_fev1": (males, "NHANES 2011–2012 males aged 6–79", "RIDAGEYR", "SPXNFEV1", "Age", "years", "FEV₁", "mL"),
    }
    out = {}
    for key, (d, desc, x, y, xl, xu, yl, yu) in pairs.items():
        s = d[[x, y]].dropna().sample(n=n, random_state=seed)
        out[key] = {
            "x": s[x].to_numpy(),
            "y": s[y].to_numpy(),
            "x_label": xl,
            "x_units": xu,
            "y_label": yl,
            "y_units": yu,
            "sample": desc,
            "population_n": int(d[[x, y]].dropna().shape[0]),
        }
    return out


def nhanes_two_feature(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Age + BMI → SBP sample used for the real decision-tree walk-through."""
    d = _nhanes_adults()
    return d[["RIDAGEYR", "BMXBMI", "SBP"]].dropna().sample(n=n, random_state=seed).reset_index(drop=True)


def all_datasets() -> list[Dataset]:
    return [nhanes_glucose(), nhanes_sbp(), nhanes_fev1(), parkinsons()]

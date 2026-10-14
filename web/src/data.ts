import { useEffect, useState } from "react";

// ---------- types mirroring pipeline/run_all.py ----------
export type ModelId = "baseline" | "linear" | "rf" | "gbm" | "tabicl";
export const MODEL_IDS: ModelId[] = ["baseline", "linear", "rf", "gbm", "tabicl"];
export const MODEL_LABEL: Record<ModelId, string> = {
  baseline: "Baseline (mean)",
  linear: "Linear",
  rf: "Random forest",
  gbm: "Gradient boosting",
  tabicl: "Foundation (TabICL)",
};
export const MODEL_COLOR: Record<ModelId, string> = {
  baseline: "var(--m-baseline)",
  linear: "var(--m-linear)",
  rf: "var(--m-rf)",
  gbm: "var(--m-gbm)",
  tabicl: "var(--m-tabicl)",
};

export type SplitId = "kfold" | "random" | "temporal" | "group";
export type DatasetId = "nhanes_glucose" | "nhanes_sbp" | "nhanes_fev1" | "parkinsons";
export const DATASET_IDS: DatasetId[] = ["nhanes_glucose", "nhanes_sbp", "nhanes_fev1", "parkinsons"];
export const DATASET_SHORT: Record<DatasetId, string> = {
  nhanes_glucose: "HbA1c → glucose",
  nhanes_sbp: "Blood pressure",
  nhanes_fev1: "Lung function (FEV₁)",
  parkinsons: "Parkinson's",
};

export interface FoldScore { train_r2: number; test_r2: number; test_rmse: number; test_mae: number; n_test: number }
export interface ModelCV {
  folds: FoldScore[];
  mean_train_r2: number; mean_test_r2: number; sd_test_r2: number;
  mean_test_rmse: number; sd_test_rmse: number; mean_test_mae: number;
  repeated_r2: number[]; repeated_rmse: number[];
}
export interface SplitCV {
  label: string; description: string; n_repeats: number;
  models: Record<ModelId, ModelCV>;
  oof: {
    fold: number[];
    pred: Record<ModelId, (number | null)[]>;
    linear_sigma: (number | null)[];
    tabicl_quantiles: (number | null)[][];
  };
}
export interface Curves { train_r2: number[]; test_r2: number[] }
export interface DatasetData {
  id: DatasetId; name: string; story: string; source: string; citation: string; url: string;
  target: string; target_label: string; units: string;
  n_rows: number; n_patients: number;
  features: { id: string; label: string }[];
  subgroup_name: string; subgroup: string[];
  y: number[];
  y_summary: { mean: number; sd: number; min: number; max: number };
  splits: SplitId[]; main_split: SplitId;
  cv: Partial<Record<SplitId, SplitCV>>;
  learning_curve: { fractions: number[]; sizes: number[]; models: Record<"linear" | "rf" | "gbm" | "tabicl", Curves> };
  boosting: { rounds: number[]; learning_rates: Record<string, Curves> };
  forest: { n_trees: number[]; min_samples_leaf: Record<string, Curves> };
  recipe: {
    intercept: number; train_r2: number;
    features: { id: string; label: string; weight: number; binary: boolean; min: number; max: number; median: number; mean: number }[];
  };
}

export interface PlaygroundPair {
  x_label: string; x_units: string; y_label: string; y_units: string; sample: string; population_n: number;
  x: number[]; y: number[]; grid: number[]; data_range: [number, number];
  linear: { intercept: number; slope: number; train_r2: number; cv_r2: number };
  rf: Record<string, { curve: number[]; train_r2: number; cv_r2: number }>;
  gbm: Record<string, { curve: number[]; train_r2: number; cv_r2: number }>;
  tabicl: { curve: number[]; lo: number[]; hi: number[]; train_r2: number; cv_r2: number };
}
export type TreeNode =
  | { leaf: true; value: number; n: number }
  | { leaf: false; feature: "age" | "bmi"; threshold: number; n: number; left: TreeNode; right: TreeNode };
export interface PlaygroundData {
  pairs: Record<"age_sbp" | "bmi_sbp" | "a1c_glucose" | "age_fev1", PlaygroundPair>;
  rf_trees: number[]; rf_leaves: number[]; gb_rounds: number[]; gb_lrs: number[];
  trees: {
    n: number; mean_sbp: number; trees: TreeNode[];
    forest_grid: { ages: number[]; bmis: number[]; pred: number[][] };
    points: { age: number[]; bmi: number[]; sbp: number[] };
  };
}

export interface LabData {
  hook: {
    patient: { age: number; bmi: number; waist: number; sex: string; income_ratio: number };
    assistant: { model: string; prediction: number; train_r2: number; gbm_train_r2: number };
    verified: { linear_prediction: number; models: Record<"rf" | "gbm" | "linear" | "baseline", { train_r2: number; cv_r2: number; cv_rmse: number }> };
    n: number; bmi_only_r2: number;
  };
  leakage: {
    runs: { keep_total: boolean; keep_id: boolean; split: "random" | "group"; train_r2: number; test_r2: number; test_r2_folds: number[]; baseline_r2: number }[];
    corr_motor_total: number; n_rows: number; n_patients: number;
  };
  nested: {
    grid: Record<string, number[]>; n_settings: number;
    best_score: number; best_params: Record<string, number>;
    nested_scores: number[]; nested_mean: number; nested_sd: number;
    outer_picks: Record<string, number>[];
    untuned_mean: number; linear_mean: number;
    all_settings: { params: Record<string, number>; score: number }[];
  };
}

export interface MetaData {
  generated: string; seed: number; n_folds: number; repeats: number; python: string;
  packages: Record<string, string>; model_details: Record<ModelId, string>; splits: Record<SplitId, string>;
  runtime_seconds: number;
}

// ---------- loader with a module-level cache ----------
const cache = new Map<string, Promise<unknown>>();

export function load<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(
      name,
      fetch(`${import.meta.env.BASE_URL}data/${name}.json`).then((r) => {
        if (!r.ok) throw new Error(`Could not load ${name}.json (${r.status})`);
        return r.json();
      }),
    );
  }
  return cache.get(name) as Promise<T>;
}

export function useData<T>(name: string | null): T | null {
  const [state, setState] = useState<{ name: string; value: T } | null>(null);
  useEffect(() => {
    if (!name) return;
    let live = true;
    load<T>(name).then((v) => live && setState({ name, value: v }));
    return () => {
      live = false;
    };
  }, [name]);
  return state && state.name === name ? state.value : null;
}

export const useDataset = (id: DatasetId | null) => useData<DatasetData>(id);

// ---------- stats helpers ----------
export const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
export const sd = (a: number[]) => {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
};
export function r2(y: number[], p: number[]) {
  const m = mean(y);
  let ssr = 0, sst = 0;
  for (let i = 0; i < y.length; i++) {
    ssr += (y[i] - p[i]) ** 2;
    sst += (y[i] - m) ** 2;
  }
  return 1 - ssr / sst;
}
export function rmse(y: number[], p: number[]) {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += (y[i] - p[i]) ** 2;
  return Math.sqrt(s / y.length);
}
export function leastSquares(x: number[], y: number[]) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < x.length; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
  }
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}
/** Least absolute deviations via iteratively reweighted least squares. */
export function leastAbsolute(x: number[], y: number[], iters = 60) {
  let { slope, intercept } = leastSquares(x, y);
  for (let k = 0; k < iters; k++) {
    let sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
    for (let i = 0; i < x.length; i++) {
      const w = 1 / Math.max(1e-6, Math.abs(y[i] - (intercept + slope * x[i])));
      sw += w; swx += w * x[i]; swy += w * y[i]; swxx += w * x[i] * x[i]; swxy += w * x[i] * y[i];
    }
    const det = sw * swxx - swx * swx;
    slope = (sw * swxy - swx * swy) / det;
    intercept = (swy - slope * swx) / sw;
  }
  return { slope, intercept };
}
/** Inverse standard normal CDF (Acklam's approximation). */
export function qnorm(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) return -qnorm(1 - p);
  const q = p - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

export const fmt = (v: number | null | undefined, d = 2) =>
  v == null || !isFinite(v) ? "–" : (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }).replace(/^-/, "−");
export const fmtSigned = (v: number, d = 2) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), d);

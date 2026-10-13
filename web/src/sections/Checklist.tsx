import { useEffect, useState } from "react";
import { Card, Callout } from "../components/ui";
import { DATASET_IDS, DatasetData, load, LabData, MetaData, useData, fmt } from "../data";

const ITEMS: { text: string; link: string }[] = [
  { text: "Split chosen from the question: KFold, GroupKFold or TimeSeriesSplit", link: "splits" },
  { text: "Every preprocessing step inside a Pipeline", link: "leakage" },
  { text: "Same folds and a fixed seed for every model", link: "compare" },
  { text: "A baseline (predict the mean) in the results table", link: "cv" },
  { text: "Mean ± SD across folds; repeated k-fold if models are close", link: "compare" },
  { text: "Nested CV if anything was tuned", link: "nested" },
  { text: "Comparison plot: per-fold or repeated-CV scores", link: "compare" },
  { text: "Diagnostics: predicted vs observed, residuals, learning curve", link: "diagnostics" },
  { text: "Subgroup check for every group the model will serve", link: "diagnostics" },
  { text: "No leaked columns: nothing derived from, or recorded after, the target", link: "leakage" },
  { text: "Test set touched once; versions + seed recorded, results saved", link: "checklist" },
];

const PROMPT = `CONTEXT  Predict motor_UPDRS (points) from voice measures. 5,875 rows from 42 patients.
TASK     Compare a mean baseline, linear regression, a tree ensemble, a tabular foundation model.
RULES    • 5-fold CV on held-out data; GroupKFold by patient if patients repeat.
         • Fit all preprocessing inside the CV loop (Pipeline).
         • Report CV R², CV RMSE (in points) and training R² side by side.
         • Drop any column derived from, or recorded after, the target.
OUTPUT   Code + table + residual plot. List every assumption you made.`;

function readChecks(): boolean[] {
  try {
    const v = JSON.parse(localStorage.getItem("checklist") || "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

export default function Checklist() {
  const [checks, setChecks] = useState<boolean[]>(readChecks);
  const lab = useData<LabData>("lab");
  const meta = useData<MetaData>("meta");
  const [sources, setSources] = useState<DatasetData[]>([]);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    Promise.all(DATASET_IDS.map((id) => load<DatasetData>(id))).then(setSources);
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem("checklist", JSON.stringify(checks));
    } catch {
      /* storage unavailable: checks still work for this visit */
    }
  }, [checks]);
  const done = ITEMS.filter((_, i) => checks[i]).length;

  return (
    <div className="stack">
      <div className="split-wide">
        <Card title={`Verification checklist · ${done} of ${ITEMS.length}`} sub="Paste it into your prompt, and attach it, completed, to your report">
          {ITEMS.map((it, i) => (
            <label key={i} className={`check-item ${checks[i] ? "done" : ""}`}>
              <input type="checkbox" checked={!!checks[i]} onChange={(e) => { const c = [...checks]; c[i] = e.target.checked; setChecks(c); }} />
              <span className="check-text">{it.text} <a href={`#/${it.link}`} className="small" onClick={(e) => e.stopPropagation()}>see →</a></span>
            </label>
          ))}
        </Card>
        <div className="stack">
          {lab && (
            <Card title="Back to our patient">
              <div className="small ink2">Age {lab.hook.patient.age}, BMI {lab.hook.patient.bmi}, {lab.hook.patient.sex.toLowerCase()}:</div>
              <div className="stat-value tnum">≈ {fmt(lab.hook.verified.linear_prediction, 0)} mmHg</div>
              <div className="small ink2">
                typically off by about <b>{fmt(lab.hook.verified.models.linear.cv_rmse, 0)} mmHg</b> on new patients (linear, held-out R² {fmt(lab.hook.verified.models.linear.cv_r2, 2)}).
                The assistant's “{fmt(lab.hook.assistant.prediction, 0)}, R² = {fmt(lab.hook.assistant.train_r2, 2)}”? Run the checklist.
              </div>
            </Card>
          )}
          <Card title="The one thing to remember">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              <li><b>Beat the baseline on unseen patients.</b> <span className="small ink2">Cross-validated, never the training score.</span></li>
              <li><b>Split the way it will be used.</b> <span className="small ink2">New patients, later visits, new sites: the question decides.</span></li>
              <li><b>Read the residuals and the subgroups.</b> <span className="small ink2">One average score can hide who the model fails.</span></li>
            </ol>
          </Card>
        </div>
      </div>

      <Card title="A reusable prompt pattern" sub="Works with any AI assistant. It writes code fast; you own the question, the split and the conclusion.">
        <pre className="code">{PROMPT}</pre>
        <button className="btn" style={{ marginTop: 10 }} onClick={() => {
          navigator.clipboard?.writeText(PROMPT).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }, () => {});
        }}>{copied ? "Copied ✓" : "Copy prompt"}</button>
      </Card>

      <Card title="Data sources and reproducibility">
        <ul className="small" style={{ paddingLeft: 18, margin: 0 }}>
          {sources.map((s) => (
            <li key={s.id} style={{ marginBottom: 6 }}>
              <b>{s.name}</b> ({s.n_rows.toLocaleString()} rows): {s.citation} <a href={s.url} target="_blank" rel="noreferrer">source</a>
            </li>
          ))}
        </ul>
        {meta && (
          <p className="small muted" style={{ marginTop: 10, marginBottom: 0 }}>
            Results generated {meta.generated.slice(0, 10)} by <code>pipeline/run_all.py</code> · seed {meta.seed} · {meta.n_folds}-fold CV × {meta.repeats} repeats ·
            Python {meta.python} · {Object.entries(meta.packages).map(([k, v]) => `${k} ${v}`).join(" · ")}.
          </p>
        )}
      </Card>
      <Callout>A model always gives you an answer, however advanced it is and whoever wrote the code. Your job is to check whether it deserves your trust.</Callout>
    </div>
  );
}

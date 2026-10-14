import { useEffect, useState } from "react";
import { TrainTestBars, TrainTestLegend } from "../charts/bars";
import { Card, Callout, DatasetPicker, Loading, Segmented, Toggle } from "../components/ui";
import { DATASET_IDS, DatasetId, fmt, MODEL_COLOR, MODEL_IDS, MODEL_LABEL, useDataset } from "../data";

const CELLS = 20;

export default function CrossValidation() {
  const [id, setId] = useState<DatasetId>("nhanes_fev1");
  const [fold, setFold] = useState(0);
  const [play, setPlay] = useState(false);
  const [metric, setMetric] = useState<"r2" | "rmse">("r2");
  const [showTrain, setShowTrain] = useState(true);
  const ds = useDataset(id);

  useEffect(() => {
    if (!play) return;
    const t = setInterval(() => setFold((f) => (f + 1) % 5), 1300);
    return () => clearInterval(t);
  }, [play]);

  const cv = ds?.cv[ds.main_split];
  return (
    <div className="stack">
      <Card title="5-fold cross-validation" sub="Every patient is tested exactly once, by a model trained on the other 80%">
        <div className="row" style={{ marginBottom: 12 }}>
          <Segmented label="Fold" value={fold} onChange={(f) => { setPlay(false); setFold(f); }} options={[0, 1, 2, 3, 4].map((f) => ({ value: f, label: `Fold ${f + 1}` }))} />
          <button className="btn" style={{ alignSelf: "flex-end" }} onClick={() => setPlay(!play)}>{play ? "❚❚ Pause" : "▶ Animate"}</button>
        </div>
        <div className="fold-grid" style={{ gridTemplateColumns: `80px repeat(${CELLS}, 1fr)` }} role="img" aria-label={`Fold ${fold + 1}: one fifth of patients held out for testing`}>
          {[0, 1, 2, 3, 4].map((f) => (
            <div key={f} style={{ display: "contents" }}>
              <div className="small" style={{ alignSelf: "center", fontWeight: f === fold ? 700 : 400, color: f === fold ? "var(--ink)" : "var(--muted)" }}>Fold {f + 1}</div>
              {Array.from({ length: CELLS }, (_, c) => {
                const test = Math.floor(c / (CELLS / 5)) === f;
                return <div key={c} className="fold-cell" style={{
                  background: test ? "var(--m-rf)" : "var(--m-linear)",
                  opacity: f === fold ? (test ? 1 : 0.75) : 0.18,
                }} />;
              })}
            </div>
          ))}
        </div>
        <div className="legend" style={{ marginTop: 10 }}>
          <span className="legend-item"><span className="swatch" style={{ background: "var(--m-rf)" }} /> Test patients (held out)</span>
          <span className="legend-item"><span className="swatch" style={{ background: "var(--m-linear)" }} /> Training patients</span>
        </div>
        {cv && (
          <div className="table-scroll" style={{ marginTop: 10 }}>
            <table className="data">
              <thead><tr><th>Fold {fold + 1} ({ds!.name})</th><th>Training R²</th><th>Held-out R²</th><th>Held-out RMSE ({ds!.units})</th></tr></thead>
              <tbody>
                {MODEL_IDS.map((m) => (
                  <tr key={m}>
                    <td><span className="swatch" style={{ background: MODEL_COLOR[m], marginRight: 8 }} />{MODEL_LABEL[m]}</td>
                    <td>{fmt(cv.models[m].folds[fold].train_r2, 3)}</td>
                    <td><b>{fmt(cv.models[m].folds[fold].test_r2, 3)}</b></td>
                    <td>{fmt(cv.models[m].folds[fold].test_rmse, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <div className="controls-bar">
        <DatasetPicker value={id} onChange={setId} ids={DATASET_IDS} />
        <Segmented label="Score" value={metric} onChange={setMetric} options={[{ value: "r2", label: "R²" }, { value: "rmse", label: "RMSE (units)" }]} />
        <Toggle checked={showTrain} onChange={setShowTrain}>Show training score</Toggle>
      </div>

      {!ds || !cv ? <Loading /> : (() => {
        const base = cv.models.baseline;
        const best = MODEL_IDS.filter((m) => m !== "baseline").reduce((b, m) => (cv.models[m].mean_test_r2 > cv.models[b].mean_test_r2 ? m : b), "linear" as typeof MODEL_IDS[number]);
        const bestM = cv.models[best];
        const gaps = MODEL_IDS.filter((m) => m !== "baseline").map((m) => ({ m, gap: cv.models[m].mean_train_r2 - cv.models[m].mean_test_r2 }));
        const worstGap = gaps.reduce((a, b) => (b.gap > a.gap ? b : a));
        return (
          <div className="split-wide">
            <Card title={`${ds.name}: mean of 5 folds`} sub={`${cv.description}`}>
              <TrainTestLegend valueLabel={metric === "r2" ? "R²" : "RMSE"} />
              <TrainTestBars
                showTrain={showTrain}
                valueLabel={metric === "r2" ? "R²" : "RMSE"}
                digits={metric === "r2" ? 2 : 1}
                domain={metric === "r2" ? undefined : [0, Math.max(...MODEL_IDS.map((m) => cv.models[m].mean_test_rmse + cv.models[m].sd_test_rmse)) * 1.05]}
                rows={MODEL_IDS.map((m) => {
                  const mm = cv.models[m];
                  const trainRmse = metric === "rmse" ? null : mm.mean_train_r2;
                  return {
                    id: m, label: MODEL_LABEL[m], color: MODEL_COLOR[m],
                    train: trainRmse,
                    test: metric === "r2" ? mm.mean_test_r2 : mm.mean_test_rmse,
                    testSd: metric === "r2" ? mm.sd_test_r2 : mm.sd_test_rmse,
                  };
                })}
              />
              {metric === "rmse" && <p className="small muted">Lower is better. RMSE is in {ds.units}; training RMSE isn't shown.</p>}
            </Card>
            <div className="stack">
              <div className="card">
                <div className="stat-label">Always include a baseline</div>
                <div className="stat-value tnum">{fmt(base.mean_test_rmse, 1)} → {fmt(bestM.mean_test_rmse, 1)}</div>
                <div className="stat-note">
                  Predicting the mean gives RMSE {fmt(base.mean_test_rmse, 1)} {ds.units} (R² ≈ 0). The best model here ({MODEL_LABEL[best]}) cuts that by{" "}
                  <b>{fmt((1 - bestM.mean_test_rmse / base.mean_test_rmse) * 100, 0)}%</b>.
                </div>
              </div>
              <div className="card">
                <div className="stat-label">The gap is overfitting</div>
                <div className="stat-value tnum" style={{ color: MODEL_COLOR[worstGap.m] }}>
                  {fmt(cv.models[worstGap.m].mean_train_r2, 2)} vs {fmt(cv.models[worstGap.m].mean_test_r2, 2)}
                </div>
                <div className="stat-note">{MODEL_LABEL[worstGap.m]}: R² on its own training patients vs on new ones.</div>
              </div>
            </div>
          </div>
        );
      })()}
      <Callout>
        In code: <code>cross_validate(model, X, y, cv=KFold(5, shuffle=True, random_state=0), return_train_score=True)</code>. Use one <code>cv</code> object for
        every model, so they are all graded on exactly the same patients.
      </Callout>
    </div>
  );
}

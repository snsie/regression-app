import { useState } from "react";
import { TrainTestBars, TrainTestLegend } from "../charts/bars";
import { Card, Callout, Loading, PauseThink, SourceNote, Toggle } from "../components/ui";
import { fmt, LabData, MODEL_COLOR, useData } from "../data";

export default function Hook() {
  const lab = useData<LabData>("lab");
  const [guess, setGuess] = useState<string>("");
  const [asked, setAsked] = useState(false);
  const [verified, setVerified] = useState(false);
  if (!lab) return <Loading />;
  const h = lab.hook;
  const v = h.verified.models;

  return (
    <div className="stack">
      <div className="grid-2">
        <Card>
          <div className="stat-label">New patient</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 650, margin: "6px 0 2px" }}>
            {h.patient.sex}, age {h.patient.age}, BMI {h.patient.bmi}
          </div>
          <div className="muted small">Waist {h.patient.waist} cm · income-to-poverty ratio {h.patient.income_ratio}</div>
          <div className="spacer" />
          <div className="stat-label">Systolic blood pressure = ?</div>
          <div className="row" style={{ marginTop: 6 }}>
            <input
              type="number" inputMode="numeric" aria-label="Your SBP guess in mmHg" placeholder="e.g. 130"
              value={guess} onChange={(e) => setGuess(e.target.value)}
            />
            <span className="muted">mmHg</span>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>
            Write your guess down before any model sees the data.
          </p>
        </Card>

        <Card>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div className="stat-label">AI assistant</div>
            <span className="chip">illustrative chat · real numbers</span>
          </div>
          <div style={{ marginTop: 10, padding: "10px 12px", borderRadius: 10, background: "var(--surface-2)" }}>
            Here's NHANES data on {h.n.toLocaleString()} adults. Predict SBP for this patient.
          </div>
          {!asked ? (
            <button className="btn primary" style={{ marginTop: 14 }} onClick={() => setAsked(true)}>
              Ask the assistant
            </button>
          ) : (
            <div style={{ marginTop: 10, padding: "12px 14px", borderRadius: 10, border: "1px solid var(--border)" }} className="pause-body">
              I trained gradient boosting and a random forest and picked the best one.
              <div style={{ fontSize: "1.6rem", fontWeight: 700, margin: "8px 0 0" }}>
                Predicted SBP: {fmt(h.assistant.prediction, 0)} mmHg
              </div>
              <div>
                Model R² = <b>{fmt(h.assistant.train_r2, 2)}</b> <span className="muted">(excellent fit)</span>
              </div>
              {guess && (
                <div className="small muted" style={{ marginTop: 6 }}>
                  Your guess: {guess} mmHg
                </div>
              )}
            </div>
          )}
        </Card>
      </div>

      {asked && (
        <PauseThink prompt="Should you believe it? List two questions you'd ask before acting on that number.">
          <ul style={{ margin: "0 0 10px", paddingLeft: 20 }}>
            <li>Was the R² measured on the same patients the model was trained on, or on new ones?</li>
            <li>How far off is a typical prediction, in mmHg? Is that better than just predicting the average?</li>
            <li>Does it work equally well for every group of patients it will be used on?</li>
          </ul>
          <Toggle checked={verified} onChange={setVerified}>
            Show the verified answer
          </Toggle>
        </PauseThink>
      )}

      {verified && (
        <Card title="Same models, graded on patients they never saw" sub="5-fold cross-validation, NHANES 2017–2018 adults">
          <TrainTestLegend sd={false} />
          <TrainTestBars
            rows={[
              { id: "rf", label: "Random forest", color: MODEL_COLOR.rf, train: v.rf.train_r2, test: v.rf.cv_r2 },
              { id: "gbm", label: "Gradient boosting", color: MODEL_COLOR.gbm, train: v.gbm.train_r2, test: v.gbm.cv_r2 },
              { id: "linear", label: "Linear", color: MODEL_COLOR.linear, train: v.linear.train_r2, test: v.linear.cv_r2 },
              { id: "baseline", label: "Baseline (mean)", color: MODEL_COLOR.baseline, train: v.baseline.train_r2, test: v.baseline.cv_r2 },
            ]}
            domain={[0, 1]}
          />
          <div className="spacer" />
          <Callout>
            The “R² = {fmt(h.assistant.train_r2, 2)}” was graded on the forest's own training patients. On new patients, the forest
            explains about <b>{fmt(v.rf.cv_r2 * 100, 0)}%</b> of the variation, and plain linear regression does slightly better (
            <b>{fmt(v.linear.cv_r2, 2)}</b>). A typical miss is about <b>±{fmt(v.linear.cv_rmse, 0)} mmHg</b>, compared with ±
            {fmt(v.baseline.cv_rmse, 0)} for guessing the average. The honest answer for this patient is about {fmt(h.verified.linear_prediction, 0)} mmHg,
            give or take {fmt(v.linear.cv_rmse, 0)}.
          </Callout>
          <SourceNote>
            The rest of this walk-through shows how to get from the first number to the second: model families, cross-validation, residuals,
            leakage and fair comparisons.
          </SourceNote>
        </Card>
      )}

      <div className="grid-4">
        {[
          ["1", "Frame regression", "See any model as y = f(x) + ε: a model family, a loss and data."],
          ["2", "Read a linear model", "Slope, intercept, residuals and least squares, in clinical units."],
          ["3", "Choose a model", "What tree ensembles and tabular foundation models add, and what they cost."],
          ["4", "Verify any model", "Baseline, held-out data, residuals, leakage, subgroups."],
        ].map(([n, t, d]) => (
          <Card key={n}>
            <div className="row" style={{ gap: 10, marginBottom: 6, flexWrap: "nowrap" }}>
              <span className="badge-num">{n}</span>
              <b>{t}</b>
            </div>
            <div className="small ink2">{d}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}

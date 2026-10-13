import { useState } from "react";
import { Card, Callout, Loading, PauseThink, Toggle } from "../components/ui";
import { fmt, LabData, useData } from "../data";

interface Fixes { dropTotal: boolean; dropId: boolean; groupSplit: boolean; heldOut: boolean; pipeline: boolean }

export default function Leakage() {
  const lab = useData<LabData>("lab");
  const [f, setF] = useState<Fixes>({ dropTotal: false, dropId: false, groupSplit: false, heldOut: false, pipeline: false });
  if (!lab) return <Loading />;
  const L = lab.leakage;
  const run = L.runs.find((r) => r.keep_total === !f.dropTotal && r.keep_id === !f.dropId && r.split === (f.groupSplit ? "group" : "random"))!;
  const score = f.heldOut ? run.test_r2 : run.train_r2;
  const allFixed = f.dropTotal && f.dropId && f.groupSplit && f.heldOut;
  const set = (k: keyof Fixes) => (v: boolean) => setF({ ...f, [k]: v });

  const cls = (flag: boolean) => (flag ? "line fixed" : "line flag");
  return (
    <div className="stack">
      <div className="split-wide">
        <Card title="What the assistant wrote" sub="Click the fixes on the right; the flagged lines change">
          <pre className="code" aria-label="Assistant-written Python code">
            <span className={cls(f.dropTotal && f.dropId)}>
              {f.dropTotal || f.dropId
                ? `X = df.drop(columns=['motor_UPDRS'${f.dropTotal ? ", 'total_UPDRS'" : ""}${f.dropId ? ", 'subject#'" : ""}])`
                : "X = df.drop(columns=['motor_UPDRS'])"}
              <span className="cm">{"   # ①②"}</span>
            </span>
            <span className="line">y = df['motor_UPDRS']</span>
            {f.pipeline
              ? <span className="line fixed">model = make_pipeline(StandardScaler(), RandomForestRegressor())<span className="cm">{"   # ④"}</span></span>
              : <span className="line flag">X = StandardScaler().fit_transform(X)<span className="cm">{"   # ④"}</span></span>}
            {f.groupSplit
              ? <span className="line fixed">cv = GroupKFold(5, shuffle=True, random_state=0)<span className="cm">{"   # ②"}</span></span>
              : <span className="line flag">cv = KFold(5, shuffle=True, random_state=42)<span className="cm">{"   # ②  random rows"}</span></span>}
            {!f.pipeline && <span className="line">model = RandomForestRegressor()</span>}
            <span className="line">r = cross_validate(model, X, y, cv=cv{f.groupSplit ? ", groups=df['subject#']" : ""},</span>
            <span className="line">                   return_train_score=True)</span>
            {f.heldOut
              ? <span className="line fixed">print('R²:', r['test_score'].mean())<span className="cm">{"   # ③"}</span></span>
              : <span className="line flag">print('R²:', r['train_score'].mean())<span className="cm">{"   # ③"}</span></span>}
          </pre>
          <div className="spacer" />
          <div className="grid-2">
            <div className="card" style={{ background: "var(--surface-2)" }}>
              <div className="stat-label">{f.heldOut ? "Reported R² (held-out)" : "Assistant reported R² (training)"}</div>
              <div className="stat-value huge tnum" style={{ color: score < 0 ? "var(--critical)" : score > 0.95 ? "var(--ink)" : "var(--accent)" }}>{fmt(score, f.heldOut ? 2 : 4)}</div>
              <div className="stat-note">{score > 0.99 ? "“Near-perfect accuracy.”" : score < 0 ? "Worse than predicting the average for everyone." : ""}</div>
            </div>
            <div className="card" style={{ background: "var(--surface-2)" }}>
              <div className="stat-label">Baseline (predict the mean)</div>
              <div className="stat-value huge tnum" style={{ color: "var(--m-baseline)" }}>{fmt(run.baseline_r2, 2)}</div>
              <div className="stat-note">Held-out R², same folds</div>
            </div>
          </div>
        </Card>

        <div className="stack">
          <Card title="Apply the fixes">
            <div className="stack" style={{ gap: 14 }}>
              <div>
                <Toggle checked={f.dropTotal} onChange={set("dropTotal")}><b>① Drop total_UPDRS</b></Toggle>
                <div className="small ink2" style={{ marginLeft: 42 }}>Target leakage: total UPDRS <i>contains</i> motor UPDRS (r = {fmt(L.corr_motor_total, 2)}).</div>
              </div>
              <div>
                <Toggle checked={f.dropId} onChange={set("dropId")}><b>② Drop patient ID</b></Toggle>
                <div className="small ink2" style={{ marginLeft: 42 }}>With a random split, the ID lets the model look up each patient's own level.</div>
              </div>
              <div>
                <Toggle checked={f.groupSplit} onChange={set("groupSplit")}><b>② Split by patient</b></Toggle>
                <div className="small ink2" style={{ marginLeft: 42 }}>{L.n_rows.toLocaleString()} recordings from {L.n_patients} patients. Random rows put the same patient on both sides.</div>
              </div>
              <div>
                <Toggle checked={f.heldOut} onChange={set("heldOut")}><b>③ Score on held-out patients</b></Toggle>
                <div className="small ink2" style={{ marginLeft: 42 }}>The original code graded the model on its own training rows.</div>
              </div>
              <div>
                <Toggle checked={f.pipeline} onChange={set("pipeline")}><b>④ Preprocess inside the pipeline</b></Toggle>
                <div className="small ink2" style={{ marginLeft: 42 }}>Harmless for trees (no change in score), but a habit that leaks with imputers, feature selection or target encoding.</div>
              </div>
            </div>
          </Card>
          <div className="row">
            <button className="btn" onClick={() => setF({ dropTotal: false, dropId: false, groupSplit: false, heldOut: false, pipeline: false })}>Reset to the assistant's code</button>
            <button className="btn primary" onClick={() => setF({ dropTotal: true, dropId: true, groupSplit: true, heldOut: true, pipeline: true })}>Apply all fixes</button>
          </div>
        </div>
      </div>

      {allFixed ? (
        <Callout tone="warn">
          Same random forest, verified: no leaked columns, graded on patients it has never seen, R² = <b>{fmt(run.test_r2, 2)}</b>. That is worse than predicting the
          average. Voice features in this dataset don't transfer to new patients. No baseline, CV, residual or subgroup check was ever offered by the assistant.
        </Callout>
      ) : (
        <PauseThink prompt="Before you toggle anything: which single fix do you think changes the score most?">
          <div className="small">
            Try each one alone. Scoring on held-out rows barely matters while the leaks are still in, because the leaked columns make held-out rows easy too.
            The score only collapses once the target leak is gone <i>and</i> patients are kept on one side of the split.
          </div>
        </PauseThink>
      )}
      <Card title="Every combination" sub="Random forest (100 trees), 5-fold CV, Parkinson's telemonitoring">
        <div className="table-scroll">
          <table className="data">
            <thead><tr><th>total_UPDRS</th><th style={{ textAlign: "left" }}>patient ID</th><th style={{ textAlign: "left" }}>Split</th><th>Training R²</th><th>Held-out R²</th></tr></thead>
            <tbody>
              {L.runs.map((r, i) => {
                const cur = r === run;
                return (
                  <tr key={i} style={cur ? { background: "var(--accent-soft)", fontWeight: 650 } : undefined}>
                    <td>{r.keep_total ? "kept (leak)" : "dropped"}</td>
                    <td style={{ textAlign: "left" }}>{r.keep_id ? "kept" : "dropped"}</td>
                    <td style={{ textAlign: "left" }}>{r.split === "random" ? "Random rows" : "By patient"}</td>
                    <td>{fmt(r.train_r2, 3)}</td>
                    <td style={{ color: r.test_r2 < 0 ? "var(--critical)" : undefined }}>{fmt(r.test_r2, 3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

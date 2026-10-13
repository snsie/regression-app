import { Plot, TipRow } from "../charts/kit";
import { Card, Callout, Loading, PauseThink, Stat } from "../components/ui";
import { fmt, LabData, MODEL_COLOR, useData } from "../data";

const pstr = (p: Record<string, number>) => `${p.num_leaves} leaves · lr ${p.learning_rate} · ${p.n_estimators} trees`;

export default function Nested() {
  const lab = useData<LabData>("lab");
  if (!lab) return <Loading />;
  const n = lab.nested;
  const sorted = [...n.all_settings].sort((a, b) => b.score - a.score);
  const picks = n.outer_picks.map(pstr);
  const distinct = new Set(picks).size;
  const changed = picks.filter((p) => p !== pstr(n.best_params)).length;
  const lo = Math.min(...sorted.map((s) => s.score), n.linear_mean) - 0.02;
  const hi = Math.max(...sorted.map((s) => s.score), n.linear_mean) + 0.02;

  return (
    <div className="stack">
      <div className="split-wide">
        <Card title="Two loops, two jobs" sub="Diabetes (442 patients), LightGBM, 24 hyperparameter settings">
          <div style={{ border: "2px solid var(--accent)", borderRadius: 12, padding: 14 }}>
            <div className="stat-label" style={{ color: "var(--accent)" }}>Outer loop · grades (5 folds)</div>
            <div className="row" style={{ gap: 4, margin: "8px 0", flexWrap: "nowrap" }}>
              {[0, 1, 2, 3, 4].map((i) => <div key={i} style={{ flex: 1, height: 22, borderRadius: 4, background: i === 4 ? "var(--m-rf)" : "var(--m-linear)", opacity: i === 4 ? 1 : 0.8 }} />)}
            </div>
            <div style={{ border: "2px dashed var(--m-gbm)", borderRadius: 10, padding: 12, marginRight: "20%" }}>
              <div className="stat-label" style={{ color: "var(--m-gbm)" }}>Inner loop · tunes, on the outer training part only</div>
              <div className="small ink2" style={{ marginTop: 4 }}>
                {n.n_settings} settings × 5 inner folds → pick the best setting, refit it, then score it <b>once</b> on the outer test fold.
              </div>
            </div>
          </div>
          <pre className="code" style={{ marginTop: 14 }}>{`inner = KFold(5, shuffle=True, random_state=1)
outer = KFold(5, shuffle=True, random_state=0)
tuned = GridSearchCV(LGBMRegressor(), grid, cv=inner)
cross_validate(tuned, X, y, cv=outer)   # ← the number to report`}</pre>
        </Card>
        <div className="stack">
          <Stat label="GridSearchCV.best_score_" value={fmt(n.best_score, 3)} note="Tuned and graded on the same folds: optimistic" color="var(--critical)" />
          <Stat label="Nested CV" value={<>{fmt(n.nested_mean, 3)} <span style={{ fontSize: "1rem" }} className="muted">± {fmt(n.nested_sd, 2)}</span></>} note="The number to report" color={MODEL_COLOR.gbm} />
          <Stat label="Optimism" value={fmt(n.best_score - n.nested_mean, 3)} note={`Small for ${n.n_settings} settings on 442 patients. It grows with bigger grids and smaller data.`} />
        </div>
      </div>

      <Card title={`All ${n.n_settings} settings, scored on the outer folds`} sub="The tallest bar is what best_score_ reports. Picking the max of 24 noisy numbers is biased upward.">
        <Plot height={300} xDomain={[-0.5, sorted.length - 0.5]} yDomain={[lo, hi]} xTicks={[]} yLabel="CV R²" ariaLabel="CV R² of each hyperparameter setting" nice={false}
          xLabel="Settings, sorted by score">
          {({ x, y, h, w, show, hide }) => {
            const bw = Math.max(4, (w / sorted.length) * 0.72);
            return (
              <g>
                {sorted.map((s, i) => (
                  <rect key={i} x={x(i) - bw / 2} y={y(s.score)} width={bw} height={Math.max(1, h - y(s.score))} rx={3}
                    fill={MODEL_COLOR.gbm} opacity={i === 0 ? 1 : 0.45}
                    onMouseEnter={() => show(x(i), y(s.score), <><TipRow label={pstr(s.params)} value="" /><TipRow label="CV R²" value={fmt(s.score, 3)} /></>)}
                    onMouseLeave={hide} />
                ))}
                <line x1={0} x2={w} y1={y(n.nested_mean)} y2={y(n.nested_mean)} stroke="var(--ink)" strokeWidth={2} strokeDasharray="6 4" />
                <text x={w - 4} y={y(n.nested_mean) - 6} textAnchor="end" fontSize={12} fill="var(--ink)">nested CV {fmt(n.nested_mean, 3)}</text>
                <line x1={0} x2={w} y1={y(n.linear_mean)} y2={y(n.linear_mean)} stroke={MODEL_COLOR.linear} strokeWidth={2} />
                <text x={w - 4} y={y(n.linear_mean) - 6} textAnchor="end" fontSize={12} fill="var(--ink)">linear regression {fmt(n.linear_mean, 3)}</text>
              </g>
            );
          }}
        </Plot>
      </Card>

      <div className="grid-2">
        <Card title="Best setting in each outer fold" sub="The “best” hyperparameters are themselves noisy">
          <table className="data">
            <thead><tr><th>Outer fold</th><th style={{ textAlign: "left" }}>Chosen setting</th><th>Held-out R²</th></tr></thead>
            <tbody>
              {n.outer_picks.map((p, i) => (
                <tr key={i}><td>{i + 1}</td><td style={{ textAlign: "left" }}>{pstr(p)}</td><td>{fmt(n.nested_scores[i], 3)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="small ink2" style={{ marginTop: 8 }}>
            {distinct} distinct setting{distinct > 1 ? "s" : ""} chosen across 5 folds; {changed} of 5 differ from the single “best” setting found without nesting.
          </p>
        </Card>
        <div className="stack">
          <Callout>
            <b>Tuning helped:</b> {fmt(n.untuned_mean, 2)} (defaults) → {fmt(n.nested_mean, 2)} (tuned, nested). But tuned boosting{" "}
            {n.nested_mean < n.linear_mean ? <>still trails linear regression ({fmt(n.linear_mean, 2)}).</> : <>now matches or beats linear regression ({fmt(n.linear_mean, 2)}).</>}{" "}
            Tuning isn't a substitute for the right model.
          </Callout>
          <PauseThink prompt="The boosting-rounds chart earlier showed a ‘best’ number of rounds. Is that peak an honest estimate?">
            <div className="small">No. It was chosen by looking at the held-out scores, so it is tuning. To report it honestly, put the choice of rounds (or early stopping) inside an inner loop.</div>
          </PauseThink>
        </div>
      </div>
    </div>
  );
}

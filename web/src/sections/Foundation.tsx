import { useMemo, useState } from "react";
import { Legend, Plot, TipRow } from "../charts/kit";
import { TrainTestBars } from "../charts/bars";
import { Card, Callout, DatasetPicker, Loading, ModelPicker, Segmented, SourceNote } from "../components/ui";
import { DATASET_IDS, DatasetId, fmt, MODEL_COLOR, MODEL_IDS, MODEL_LABEL, qnorm, useDataset } from "../data";

// TabICL quantile columns written by the pipeline
const Q = [0.05, 0.1, 0.25, 0.75, 0.9, 0.95];
const LEVELS = [0.5, 0.8, 0.9];

export default function Foundation() {
  const [id, setId] = useState<DatasetId>("diabetes");
  const [level, setLevel] = useState(0.9);
  const [model, setModel] = useState<"tabicl" | "linear">("tabicl");
  const ds = useDataset(id);

  const calc = useMemo(() => {
    if (!ds) return null;
    const cv = ds.cv[ds.main_split]!;
    const y = ds.y;
    const lo_q = (1 - level) / 2, hi_q = 1 - lo_q;
    const li = Q.findIndex((q) => Math.abs(q - lo_q) < 1e-6), hi = Q.findIndex((q) => Math.abs(q - hi_q) < 1e-6);
    const z = qnorm(hi_q);
    const rows: { i: number; y: number; m: number; lo: number; hi: number }[] = [];
    for (let i = 0; i < y.length; i++) {
      if (model === "tabicl") {
        const q = cv.oof.tabicl_quantiles[i];
        const m = cv.oof.pred.tabicl[i];
        if (!q || q[li] == null || m == null) continue;
        rows.push({ i, y: y[i], m, lo: q[li]!, hi: q[hi]! });
      } else {
        const m = cv.oof.pred.linear[i], s = cv.oof.linear_sigma[i];
        if (m == null || s == null) continue;
        rows.push({ i, y: y[i], m, lo: m - z * s, hi: m + z * s });
      }
    }
    const inside = rows.filter((r) => r.y >= r.lo && r.y <= r.hi).length;
    const width = rows.reduce((s, r) => s + (r.hi - r.lo), 0) / rows.length;
    // a deterministic, evenly spread sample of 40 patients sorted by prediction
    const sorted = [...rows].sort((a, b) => a.m - b.m);
    const step = Math.max(1, Math.floor(sorted.length / 40));
    const sample = sorted.filter((_, k) => k % step === 0).slice(0, 40);
    return { coverage: inside / rows.length, width, n: rows.length, sample, cv };
  }, [ds, level, model]);

  if (!ds || !calc) return <Loading />;
  const s = calc.sample;
  const ylo = Math.min(...s.map((r) => Math.min(r.lo, r.y))), yhi = Math.max(...s.map((r) => Math.max(r.hi, r.y)));
  const good = Math.abs(calc.coverage - level) <= 0.03;

  return (
    <div className="stack">
      <div className="grid-3">
        {[
          ["Practice", "It practised regression on millions of synthetic tables with known answers, before ever seeing yours."],
          ["Read", "Your training rows go in as the prompt, like worked examples. No training loop, no tuning."],
          ["Answer", "One forward pass returns a prediction and a full distribution: how unsure it is."],
        ].map(([t, d], i) => (
          <Card key={t}>
            <div className="row" style={{ gap: 8, marginBottom: 6 }}>
              <span className="badge-num" style={{ background: MODEL_COLOR.tabicl }}>{i + 1}</span><b>{t}</b>
            </div>
            <div className="small ink2">{d}</div>
          </Card>
        ))}
      </div>

      <div className="controls-bar">
        <DatasetPicker value={id} onChange={setId} ids={DATASET_IDS} />
        <ModelPicker value={model} onChange={(m) => setModel(m as "tabicl" | "linear")} models={["tabicl", "linear"]} label="Intervals from" />
        <Segmented label="Nominal interval" value={level} onChange={setLevel} options={LEVELS.map((l) => ({ value: l, label: `${l * 100}%` }))} />
      </div>

      <div className="split-wide">
        <Card title={`${level * 100}% prediction intervals for 40 held-out patients`} sub={`${ds.name} · sorted by prediction · ${calc.cv.label} split`}>
          <Legend items={[
            { label: "Interval", color: MODEL_COLOR[model], kind: "box" },
            { label: "Observed: inside", color: "var(--ink)", kind: "dot" },
            { label: "Observed: outside", color: "var(--critical)", kind: "dot" },
          ]} />
          <Plot height={360} xDomain={[-1, s.length]} yDomain={[ylo, yhi]} xTicks={[]} nice
            xLabel="Held-out patients, sorted by prediction" yLabel={`${ds.target_label} (${ds.units})`} ariaLabel="Prediction intervals and observed values">
            {({ x, y, show, hide }) => (
              <g>
                {s.map((r, k) => {
                  const ok = r.y >= r.lo && r.y <= r.hi;
                  return (
                    <g key={r.i} onMouseEnter={() => show(x(k), y(r.hi), (
                      <>
                        <TipRow label="Prediction" value={fmt(r.m, 1)} />
                        <TipRow label={`${level * 100}% interval`} value={`${fmt(r.lo, 0)} – ${fmt(r.hi, 0)}`} />
                        <TipRow label="Observed" value={fmt(r.y, 1)} />
                      </>
                    ))} onMouseLeave={hide}>
                      <rect x={x(k) - 5} y={y(r.hi)} width={10} height={Math.max(1, y(r.lo) - y(r.hi))} rx={4} fill={MODEL_COLOR[model]} opacity={0.28} />
                      <line x1={x(k) - 5} x2={x(k) + 5} y1={y(r.m)} y2={y(r.m)} stroke={MODEL_COLOR[model]} strokeWidth={2.5} />
                      <circle cx={x(k)} cy={y(r.y)} r={4} fill={ok ? "var(--ink)" : "var(--critical)"} stroke="var(--surface)" strokeWidth={1.2} />
                    </g>
                  );
                })}
              </g>
            )}
          </Plot>
        </Card>

        <div className="stack">
          <div className="card">
            <div className="stat-label">Held-out patients inside the “{level * 100}%” interval</div>
            <div className="stat-value huge tnum">{fmt(calc.coverage * 100, 0)}%</div>
            <div className="stat-note">
              {good ? "✓ Close to nominal: the stated uncertainty is honest." : calc.coverage < level ? "⚠ Below nominal: the intervals are too narrow (over-confident)." : "⚠ Above nominal: the intervals are wider than they need to be."}
              {" "}All {calc.n.toLocaleString()} out-of-fold predictions. Average width {fmt(calc.width, 1)} {ds.units}.
            </div>
          </div>
          <Card title="Accuracy on the same folds" sub="Held-out R², mean of 5 folds">
            <TrainTestBars showTrain={false} rowHeight={34} labelWidth={130}
              rows={MODEL_IDS.map((m) => ({ id: m, label: MODEL_LABEL[m], color: MODEL_COLOR[m], test: calc.cv.models[m].mean_test_r2, testSd: calc.cv.models[m].sd_test_r2 }))} />
          </Card>
        </div>
      </div>

      <h2>Two foundation models, one caveat</h2>
      <div className="grid-2">
        <Card title="TabICL v2 · used in this app" sub="Open source · Apache-2.0">
          <div className="small ink2">
            In-context learning with fully open weights. It supports regression and quantile output. A good default when licence terms matter.
            <pre className="code" style={{ marginTop: 10 }}>pip install tabicl{"\n"}TabICLRegressor().fit(X_train, y_train).predict(X_test)</pre>
          </div>
        </Card>
        <Card title="TabPFN" sub="Prior Labs">
          <div className="small ink2">
            Same idea, up to about a million rows. The code is open, but the latest model weights are for non-commercial research only and need a licence
            token. Commercial use needs a licence. Check the current terms before using it in a course or product.
            <pre className="code" style={{ marginTop: 10 }}>pip install tabpfn</pre>
          </div>
        </Card>
      </div>
      <Callout tone="warn">
        <b>Check the source.</b> Vendor technical reports often rank their own model first, and benchmark authors may come from the same lab. A benchmark is a
        claim: verify it on <i>your</i> data, on the same folds as every other model, and check whether its intervals are calibrated, as above.
      </Callout>
      <SourceNote>
        In the clinic: TabPFN predicted toric ICL lens rotation after surgery (642 eyes from 371 patients; RMSE 10.7°). Ask: were both eyes of a patient kept on
        the same side of the split? Jiang Y et al. Transl Vis Sci Technol 2025;14(11):12. Interval for linear regression: ŷ ± z·σ, where σ is the training-fold
        residual SD. TabICL intervals are its own predicted quantiles.
      </SourceNote>
    </div>
  );
}

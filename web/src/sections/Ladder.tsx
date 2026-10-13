import { useState } from "react";
import { area, line } from "d3-shape";
import { Plot, TipRow, Legend, HoverColumns } from "../charts/kit";
import { TrainTestBars, TrainTestLegend } from "../charts/bars";
import { Card, Callout, Loading, Segmented, StepSlider, Toggle } from "../components/ui";
import { fmt, mean, MODEL_COLOR, ModelId, PlaygroundData, useData } from "../data";
import { PAIR_OPTIONS, PairId } from "./Framing";

const RUNGS: { id: ModelId; n: number; name: string; adds: string; costs: string; tools: string }[] = [
  { id: "baseline", n: 0, name: "Baseline", adds: "The average, for everyone", costs: "Nothing: it's the bar every model must beat", tools: "DummyRegressor" },
  { id: "linear", n: 1, name: "Linear", adds: "Straight-line effects; one readable equation", costs: "Curves and interactions", tools: "scikit-learn LinearRegression" },
  { id: "rf", n: 2, name: "Random forest", adds: "Curves and interactions; averages hundreds of trees", costs: "A single equation; can memorise", tools: "scikit-learn" },
  { id: "gbm", n: 3, name: "Gradient boosting", adds: "Each tree fixes the last one's residuals", costs: "Needs tuning; overfits small data", tools: "LightGBM · XGBoost · CatBoost" },
  { id: "tabicl", n: 4, name: "Tabular foundation model", adds: "Pretrained on millions of synthetic tables; no training loop", costs: "Large download; licence terms; compute", tools: "TabICL · TabPFN" },
];

export default function Ladder() {
  const pg = useData<PlaygroundData>("playground");
  const [pair, setPair] = useState<PairId>("age_sbp");
  const [on, setOn] = useState<Record<ModelId, boolean>>({ baseline: true, linear: true, rf: true, gbm: false, tabicl: false });
  const [trees, setTrees] = useState(100);
  const [leaf, setLeaf] = useState(1);
  const [lr, setLr] = useState(0.1);
  const [rounds, setRounds] = useState(50);
  const [band, setBand] = useState(true);
  const [extrap, setExtrap] = useState(true);
  if (!pg) return <Loading />;
  const p = pg.pairs[pair];
  const g = p.grid;
  const ybar = mean(p.y);
  const lin = g.map((v) => p.linear.intercept + p.linear.slope * v);
  const rf = p.rf[`${trees}_${leaf}`];
  const gb = p.gbm[`${lr}_${rounds}`];
  const curves: Partial<Record<ModelId, number[]>> = {
    baseline: g.map(() => ybar), linear: lin, rf: rf.curve, gbm: gb.curve, tabicl: p.tabicl.curve,
  };
  const scores: Record<ModelId, { train: number; cv: number }> = {
    baseline: { train: 0, cv: p.gbm[`${lr}_0`].cv_r2 },
    linear: { train: p.linear.train_r2, cv: p.linear.cv_r2 },
    rf: { train: rf.train_r2, cv: rf.cv_r2 },
    gbm: { train: gb.train_r2, cv: gb.cv_r2 },
    tabicl: { train: p.tabicl.train_r2, cv: p.tabicl.cv_r2 },
  };
  const ys = [...p.y];
  const ylo = Math.min(...ys), yhi = Math.max(...ys);
  const pad = (yhi - ylo) * 0.08;
  const shown = (Object.keys(on) as ModelId[]).filter((m) => on[m]);

  return (
    <div className="stack">
      <div className="grid-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))" }}>
        {RUNGS.map((r) => (
          <button key={r.id} className="card card-btn" onClick={() => setOn({ ...on, [r.id]: !on[r.id] })} aria-pressed={on[r.id]}
            style={{ textAlign: "left", cursor: "pointer", borderColor: on[r.id] ? MODEL_COLOR[r.id] : undefined, borderWidth: on[r.id] ? 2 : 1, opacity: on[r.id] ? 1 : 0.7 }}>
            <div className="row" style={{ gap: 8 }}>
              <span className="badge-num" style={{ background: MODEL_COLOR[r.id] }}>{r.n}</span>
              <b>{r.name}</b>
            </div>
            <div className="small" style={{ marginTop: 8 }}><b className="muted">ADDS</b> {r.adds}</div>
            <div className="small" style={{ marginTop: 4 }}><b className="muted">COSTS</b> {r.costs}</div>
            <div className="small muted" style={{ marginTop: 6 }}>{r.tools}</div>
          </button>
        ))}
      </div>
      <p className="small muted" style={{ margin: 0 }}>More flexible and less transparent as you go right. Click a rung to show or hide it on the chart.</p>

      <div className="split-wide">
        <Card title={`${p.y_label} vs ${p.x_label}: each model's f(x)`} sub="300 NHANES adults. Lines show each model's prediction across the range, and beyond it.">
          <div className="controls-bar">
            <Segmented label="Data" value={pair} onChange={setPair} options={PAIR_OPTIONS} />
            <Toggle checked={extrap} onChange={setExtrap}>Show beyond the data</Toggle>
            {on.tabicl && <Toggle checked={band} onChange={setBand}>Foundation model 90% interval</Toggle>}
          </div>
          <Legend items={shown.map((m) => ({ label: RUNGS.find((r) => r.id === m)!.name, color: MODEL_COLOR[m] }))} />
          <Plot height={420}
            xDomain={extrap ? [g[0], g[g.length - 1]] : p.data_range} yDomain={[ylo - pad, yhi + pad]} nice={false}
            xLabel={`${p.x_label} (${p.x_units})`} yLabel={`${p.y_label} (${p.y_units})`}
            ariaLabel="Model predictions overlaid on patient data">
            {(ctx) => {
              const { x, y, w, h } = ctx;
              const mk = line<number>().x((_, i) => x(g[i])).y((v) => y(v));
              const visIdx = g.map((v, i) => (extrap || (v >= p.data_range[0] && v <= p.data_range[1]) ? i : -1)).filter((i) => i >= 0);
              const clipId = `clip-${pair}`;
              return (
                <g>
                  <defs><clipPath id={clipId}><rect width={w} height={h} /></clipPath></defs>
                  {extrap && (
                    <>
                      <rect x={0} y={0} width={Math.max(0, x(p.data_range[0]))} height={h} fill="var(--surface-2)" />
                      <rect x={x(p.data_range[1])} y={0} width={Math.max(0, w - x(p.data_range[1]))} height={h} fill="var(--surface-2)" />
                      <text x={6} y={16} fontSize={11.5} fill="var(--muted)">no data here</text>
                      <text x={w - 6} y={16} fontSize={11.5} fill="var(--muted)" textAnchor="end">no data here</text>
                    </>
                  )}
                  {p.x.map((xv, i) => (
                    <circle key={i} cx={x(xv)} cy={y(p.y[i])} r={3.2} fill="var(--muted)" opacity={0.35} />
                  ))}
                  <g clipPath={`url(#${clipId})`}>
                    {on.tabicl && band && (
                      <path d={area<number>().x((i) => x(g[i])).y0((i) => y(p.tabicl.lo[i])).y1((i) => y(p.tabicl.hi[i]))(visIdx) ?? ""}
                        fill="var(--m-tabicl)" opacity={0.13} />
                    )}
                    {shown.map((m) => {
                      const c = curves[m]!;
                      const d = line<number>().x((i) => x(g[i])).y((i) => y(c[i]))(visIdx);
                      return <path key={m} d={d ?? mk(c) ?? ""} fill="none" stroke={MODEL_COLOR[m]} strokeWidth={m === "baseline" ? 2 : 2.5}
                        strokeDasharray={m === "baseline" ? "6 4" : undefined} strokeLinejoin="round" />;
                    })}
                  </g>
                  <HoverColumns xs={visIdx.map((i) => g[i])} ctx={ctx} render={(k) => {
                    const i = visIdx[k];
                    return (
                      <>
                        <TipRow label={<b>{p.x_label} {fmt(g[i], 1)}</b>} value="" />
                        {shown.map((m) => <TipRow key={m} color={MODEL_COLOR[m]} label={RUNGS.find((r) => r.id === m)!.name} value={fmt(curves[m]![i], 1)} />)}
                      </>
                    );
                  }} />
                </g>
              );
            }}
          </Plot>
        </Card>

        <div className="stack">
          {(on.rf || on.gbm) && (
            <Card title="Flexibility knobs">
              {on.rf && (
                <div className="stack" style={{ gap: 6, marginBottom: 12 }}>
                  <div className="row" style={{ gap: 6 }}><span className="swatch" style={{ background: MODEL_COLOR.rf }} /><b className="small">Random forest</b></div>
                  <StepSlider label="Number of trees" values={pg.rf_trees} value={trees} onChange={setTrees} />
                  <Segmented label="Min. patients per leaf" value={leaf} onChange={setLeaf} options={pg.rf_leaves.map((v) => ({ value: v, label: String(v) }))} />
                </div>
              )}
              {on.gbm && (
                <div className="stack" style={{ gap: 6 }}>
                  <div className="row" style={{ gap: 6 }}><span className="swatch" style={{ background: MODEL_COLOR.gbm }} /><b className="small">Gradient boosting</b></div>
                  <StepSlider label="Boosting rounds (trees)" values={pg.gb_rounds} value={rounds} onChange={setRounds} />
                  <Segmented label="Learning rate" value={lr} onChange={setLr} options={pg.gb_lrs.map((v) => ({ value: v, label: String(v) }))} />
                </div>
              )}
            </Card>
          )}
          <Card title="Training vs held-out R²" sub="5-fold CV on these 300 patients, for the current settings">
            <TrainTestLegend sd={false} />
            <TrainTestBars rowHeight={38} labelWidth={120}
              rows={shown.map((m) => ({ id: m, label: RUNGS.find((r) => r.id === m)!.name.replace("Tabular foundation model", "Foundation"), color: MODEL_COLOR[m], train: scores[m].train, test: scores[m].cv }))} />
          </Card>
        </div>
      </div>
      <Callout>
        Try it: random forest with 1 patient per leaf on <b>BMI → SBP</b>. The line chases individual patients, training R² climbs, and held-out R² goes
        <i> negative</i>: worse than the baseline. Then switch on <b>Show beyond the data</b>. Trees go flat at the edge of the data, while the line keeps going
        confidently. Neither is safe outside the range you trained on.
      </Callout>
    </div>
  );
}

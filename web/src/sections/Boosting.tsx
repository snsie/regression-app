import { useState } from "react";
import { line } from "d3-shape";
import { HoverColumns, Legend, Plot, TipRow } from "../charts/kit";
import { Card, Callout, DatasetPicker, Loading, Segmented, Slider, SourceNote, StepSlider } from "../components/ui";
import { DATASET_IDS, DatasetId, fmt, PlaygroundData, useData, useDataset } from "../data";
import { PAIR_OPTIONS, PairId } from "./Framing";

function interp(grid: number[], vals: number[], x: number) {
  if (x <= grid[0]) return vals[0];
  for (let i = 1; i < grid.length; i++) {
    if (x <= grid[i]) {
      const t = (x - grid[i - 1]) / (grid[i] - grid[i - 1]);
      return vals[i - 1] + t * (vals[i] - vals[i - 1]);
    }
  }
  return vals[vals.length - 1];
}

export default function Boosting() {
  const pg = useData<PlaygroundData>("playground");
  const [pair, setPair] = useState<PairId>("age_sbp");
  const [rounds, setRounds] = useState(3);
  const [lr, setLr] = useState(0.3);
  const [px, setPx] = useState<number | null>(null);
  const [dsId, setDsId] = useState<DatasetId>("nhanes_sbp");
  const [dlr, setDlr] = useState("0.1");
  const ds = useDataset(dsId);
  if (!pg) return <Loading />;
  const p = pg.pairs[pair];
  const g = p.grid;
  const [x0, x1] = p.data_range;
  const xPatient = px != null && px >= x0 && px <= x1 ? px : (x0 + x1) / 2;
  const curve = p.gbm[`${lr}_${rounds}`];
  const prevIdx = Math.max(0, pg.gb_rounds.indexOf(rounds) - 1);
  const prev = p.gbm[`${lr}_${pg.gb_rounds[prevIdx]}`];
  const inRange = g.map((v, i) => (v >= x0 && v <= x1 ? i : -1)).filter((i) => i >= 0);
  const ys = p.y;
  const ylo = Math.min(...ys), yhi = Math.max(...ys), pad = (yhi - ylo) * 0.06;
  const steps = pg.gb_rounds.filter((r) => r <= rounds).map((r) => ({ r, v: interp(g, p.gbm[`${lr}_${r}`].curve, xPatient) }));

  return (
    <div className="stack">
      <div className="split-wide">
        <Card title="Watch the trees add up" sub={`Small trees (4 leaves each) on 300 ${p.sample}`}>
          <div className="controls-bar">
            <Segmented label="Data" value={pair} onChange={(v) => { setPair(v); setPx(null); }} options={PAIR_OPTIONS} />
            <Segmented label="Learning rate" value={lr} onChange={setLr} options={pg.gb_lrs.map((v) => ({ value: v, label: String(v) }))} />
          </div>
          <div className="controls-bar">
            <StepSlider label="Boosting rounds" values={pg.gb_rounds} value={rounds} onChange={setRounds} />
            <Slider label={`Follow one patient: ${p.x_label}`} value={Math.round(xPatient * 10) / 10} min={Math.ceil(x0)} max={Math.floor(x1)} step={0.1} onChange={setPx} format={(v) => `${fmt(v, 1)} ${p.x_units}`} />
          </div>
          <Legend items={[
            { label: `After ${rounds} round${rounds === 1 ? "" : "s"}`, color: "var(--m-gbm)" },
            { label: "One round earlier", color: "var(--m-gbm)", kind: "dash" },
          ]} />
          <Plot height={380} xDomain={[x0, x1]} yDomain={[ylo - pad, yhi + pad]} nice={false}
            xLabel={`${p.x_label} (${p.x_units})`} yLabel={`${p.y_label} (${p.y_units})`} ariaLabel="Boosted model prediction curve">
            {({ x, y }) => {
              const mk = (c: number[]) => line<number>().x((i) => x(g[i])).y((i) => y(c[i]))(inRange) ?? "";
              const yp = interp(g, curve.curve, xPatient);
              return (
                <g>
                  {p.x.map((xv, i) => <circle key={i} cx={x(xv)} cy={y(p.y[i])} r={3.2} fill="var(--muted)" opacity={0.35} />)}
                  {rounds > 0 && <path d={mk(prev.curve)} fill="none" stroke="var(--m-gbm)" strokeWidth={1.8} strokeDasharray="5 4" opacity={0.6} />}
                  <path d={mk(curve.curve)} fill="none" stroke="var(--m-gbm)" strokeWidth={2.8} />
                  <line x1={x(xPatient)} x2={x(xPatient)} y1={0} y2={y(ylo - pad)} stroke="var(--axis)" strokeDasharray="3 3" />
                  <circle cx={x(xPatient)} cy={y(yp)} r={7} fill="var(--m-gbm)" stroke="var(--surface)" strokeWidth={2} />
                </g>
              );
            }}
          </Plot>
        </Card>

        <div className="stack">
          <Card title="This patient, round by round" sub={`${p.x_label} ${fmt(xPatient, 1)} ${p.x_units}, learning rate ${lr}`}>
            <div className="stack" style={{ gap: 4, maxHeight: 300, overflowY: "auto" }}>
              {steps.map((s, i) => {
                const d = i === 0 ? null : s.v - steps[i - 1].v;
                return (
                  <div key={s.r} className="row tnum small" style={{ justifyContent: "space-between", padding: "4px 8px", borderRadius: 6, background: i === steps.length - 1 ? "var(--accent-soft)" : undefined }}>
                    <span>{s.r === 0 ? "Start: the average" : `After round ${s.r}`}</span>
                    <span>
                      {d != null && <span className="muted" style={{ marginRight: 10 }}>{d >= 0 ? "+" : "−"}{fmt(Math.abs(d), 1)}</span>}
                      <b>{fmt(s.v, 1)}</b>
                    </span>
                  </div>
                );
              })}
            </div>
          </Card>
          <Card>
            <ol style={{ margin: 0, paddingLeft: 20 }} className="small">
              <li>Start simple: predict the average for everyone, then look at what's still wrong (the residuals).</li>
              <li>Fit a small tree to those residuals and add its correction, scaled down by the <b>learning rate</b>.</li>
              <li>Repeat hundreds of times. Errors shrink until the trees start fitting noise: <b>overfitting</b>.</li>
            </ol>
          </Card>
        </div>
      </div>

      <Card title="When does it start fitting noise?" sub="LightGBM defaults (31 leaves per tree), 5-fold CV. Same folds for every point.">
        <div className="controls-bar">
          <DatasetPicker value={dsId} onChange={setDsId} ids={DATASET_IDS} />
          <Segmented label="Learning rate" value={dlr} onChange={setDlr} options={["0.01", "0.05", "0.1", "0.3"].map((v) => ({ value: v, label: v }))} />
        </div>
        {!ds ? <Loading /> : (() => {
          const b = ds.boosting;
          const c = b.learning_rates[dlr];
          const best = c.test_r2.reduce((bi, v, i) => (v > c.test_r2[bi] ? i : bi), 0);
          const at100 = b.rounds.indexOf(100);
          const lo = Math.min(0, ...c.test_r2);
          return (
            <>
              <Legend items={[{ label: "Training R²", color: "var(--m-gbm)", kind: "dash" }, { label: "Held-out (CV) R²", color: "var(--m-gbm)" }]} />
              <Plot height={320} xDomain={[1, 500]} xLog yDomain={[lo, 1]} xTicks={[1, 2, 5, 10, 20, 50, 100, 200, 500]}
                xLabel="Boosting rounds (log scale)" yLabel="R²" ariaLabel="Training and held-out R² by boosting round" zeroLine>
                {(ctx) => {
                  const { x, y } = ctx;
                  const mk = (v: number[]) => line<number>().x((_, i) => x(b.rounds[i])).y((d) => y(d))(v) ?? "";
                  return (
                    <g>
                      <path d={mk(c.train_r2)} fill="none" stroke="var(--m-gbm)" strokeWidth={2} strokeDasharray="6 4" />
                      <path d={mk(c.test_r2)} fill="none" stroke="var(--m-gbm)" strokeWidth={2.8} />
                      <circle cx={x(b.rounds[best])} cy={y(c.test_r2[best])} r={6} fill="var(--m-gbm)" stroke="var(--surface)" strokeWidth={2} />
                      <text x={x(b.rounds[best])} y={y(c.test_r2[best]) - 12} textAnchor="middle" fontSize={12} fill="var(--ink)">best: {b.rounds[best]} rounds</text>
                      {at100 >= 0 && <line x1={x(100)} x2={x(100)} y1={0} y2={ctx.h} stroke="var(--axis)" strokeDasharray="2 3" />}
                      {at100 >= 0 && <text x={x(100) + 4} y={ctx.h - 6} fontSize={11} fill="var(--muted)">default</text>}
                      <HoverColumns xs={b.rounds} ctx={ctx} render={(i) => (
                        <>
                          <TipRow label={<b>{b.rounds[i]} rounds</b>} value="" />
                          <TipRow label="Training R²" value={fmt(c.train_r2[i], 3)} />
                          <TipRow label="Held-out R²" value={fmt(c.test_r2[i], 3)} />
                        </>
                      )} />
                    </g>
                  );
                }}
              </Plot>
              <p className="small ink2" style={{ marginTop: 8 }}>
                At the default 100 rounds: training R² <b>{fmt(c.train_r2[at100], 2)}</b> vs held-out <b>{fmt(c.test_r2[at100], 2)}</b>.
                The gap is overfitting. Peak held-out R² {fmt(c.test_r2[best], 2)} at {b.rounds[best]} rounds. (Picking the round count this way is itself tuning:
                see <a href="#/nested">nested CV</a>.)
              </p>
            </>
          );
        })()}
      </Card>
      <div className="grid-2">
        <Callout tone="good"><b>Reach for it when</b> you have many rows of tabular data and want top accuracy. It's the long-standing default for tables.</Callout>
        <Callout tone="warn"><b>Watch out for</b> overfitting small data and the many settings to tune, honestly.</Callout>
      </div>
      <SourceNote>
        In the clinic: XGBoost predicted operating-room case time for 46,986 surgeries. The share of cases predicted within ±10% rose from 32%
        (scheduler estimate) to 39%. Single centre, one held-out split. Bartek MA et al. J Am Coll Surg 2019;229:346–54.
      </SourceNote>
    </div>
  );
}

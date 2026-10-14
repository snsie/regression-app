import { useMemo, useState } from "react";
import { line } from "d3-shape";
import { extent, HoverColumns, Legend, Plot, TipRow } from "../charts/kit";
import { TrainTestBars } from "../charts/bars";
import { Card, Callout, DatasetPicker, Loading, ModelPicker, Segmented, SourceNote } from "../components/ui";
import { DATASET_IDS, DatasetId, fmt, MODEL_COLOR, MODEL_LABEL, ModelId, r2, rmse, SplitId, useDataset } from "../data";

const MAX_POINTS = 1500;

export default function Diagnostics() {
  const [id, setId] = useState<DatasetId>("nhanes_fev1");
  const [model, setModel] = useState<ModelId>("linear");
  const [split, setSplit] = useState<SplitId | null>(null);
  const ds = useDataset(id);
  const sp: SplitId | null = ds ? (split && ds.splits.includes(split) ? split : ds.main_split) : null;

  const d = useMemo(() => {
    if (!ds || !sp) return null;
    const cv = ds.cv[sp]!;
    const pred = cv.oof.pred[model];
    const rows: { y: number; p: number; g: string }[] = [];
    ds.y.forEach((y, i) => {
      const p = pred[i];
      if (p != null) rows.push({ y, p, g: ds.subgroup[i] });
    });
    const step = Math.max(1, Math.ceil(rows.length / MAX_POINTS));
    const shown = rows.filter((_, i) => i % step === 0);
    // binned mean residual vs prediction (12 equal-count bins) as a trend line
    const sorted = [...rows].sort((a, b) => a.p - b.p);
    const nb = 12, bins: { p: number; r: number }[] = [];
    for (let b = 0; b < nb; b++) {
      const seg = sorted.slice(Math.floor((b * sorted.length) / nb), Math.floor(((b + 1) * sorted.length) / nb));
      if (seg.length) bins.push({ p: seg.reduce((s, r) => s + r.p, 0) / seg.length, r: seg.reduce((s, r) => s + (r.y - r.p), 0) / seg.length });
    }
    const groups = [...new Set(rows.map((r) => r.g))].sort();
    const sub = groups.map((g) => {
      const rr = rows.filter((r) => r.g === g);
      return { g, n: rr.length, rmse: rmse(rr.map((r) => r.y), rr.map((r) => r.p)), r2: r2(rr.map((r) => r.y), rr.map((r) => r.p)) };
    }).filter((s) => s.n >= 20);
    return { rows, shown, bins, sub, cv, step };
  }, [ds, sp, model]);

  if (!ds || !d || !sp) return <Loading />;
  const all = d.rows.flatMap((r) => [r.y, r.p]);
  const dom = extent(all, 0.03);
  const resid = d.rows.map((r) => r.y - r.p);
  const rl = Math.max(...resid.map(Math.abs)) * 1.05;
  const lc = ds.learning_curve;
  const lcModel = (model === "baseline" ? "linear" : model) as "linear" | "rf" | "gbm" | "tabicl";
  const lcd = lc.models[lcModel];
  const color = MODEL_COLOR[model];
  const finalGap = lcd.train_r2[lcd.train_r2.length - 1] - lcd.test_r2[lcd.test_r2.length - 1];

  return (
    <div className="stack">
      <div className="controls-bar">
        <DatasetPicker value={id} onChange={(v) => { setId(v); setSplit(null); }} ids={DATASET_IDS} />
        <ModelPicker value={model} onChange={setModel} models={["linear", "rf", "gbm", "tabicl"]} />
        {ds.splits.length > 1 && (
          <Segmented label="Split" value={sp} onChange={setSplit} options={ds.splits.map((s) => ({ value: s, label: ds.cv[s]!.label }))} />
        )}
      </div>
      <p className="small muted" style={{ margin: 0 }}>
        All plots use out-of-fold predictions: each patient is predicted by a model that never saw them.
        {d.step > 1 && ` Showing every ${d.step}th of ${d.rows.length.toLocaleString()} points.`}
      </p>

      <div className="grid-2">
        <Card title="Predicted vs observed" sub="Healthy: points hug the diagonal across the whole range">
          <Plot height={340} xDomain={dom} yDomain={dom} xLabel={`Predicted (${ds.units})`} yLabel={`Observed (${ds.units})`} ariaLabel="Predicted versus observed scatter">
            {({ x, y }) => (
              <g>
                <line x1={x(dom[0])} y1={y(dom[0])} x2={x(dom[1])} y2={y(dom[1])} stroke="var(--ink-2)" strokeDasharray="5 4" />
                {d.shown.map((r, i) => <circle key={i} cx={x(r.p)} cy={y(r.y)} r={2.8} fill={color} opacity={0.4} />)}
              </g>
            )}
          </Plot>
          <p className="small ink2">Warning sign: a <b>tilted cloud</b>. High values under-predicted and low values over-predicted means the model shrinks toward the mean.</p>
        </Card>

        <Card title="Residuals vs predicted" sub="Healthy: a shapeless band around zero">
          <Legend items={[{ label: "Residual (observed − predicted)", color, kind: "dot" }, { label: "Average residual (binned)", color: "var(--ink)" }]} />
          <Plot height={316} xDomain={dom} yDomain={[-rl, rl]} xLabel={`Predicted (${ds.units})`} yLabel="Residual" ariaLabel="Residual plot" zeroLine>
            {({ x, y }) => (
              <g>
                {d.shown.map((r, i) => <circle key={i} cx={x(r.p)} cy={y(r.y - r.p)} r={2.8} fill={color} opacity={0.35} />)}
                <path d={line<{ p: number; r: number }>().x((b) => x(b.p)).y((b) => y(b.r))(d.bins) ?? ""} fill="none" stroke="var(--ink)" strokeWidth={2.5} />
              </g>
            )}
          </Plot>
          <p className="small ink2">Warning signs: a <b>curve</b> (a missing nonlinearity) or a <b>funnel</b> (error grows with the prediction).</p>
        </Card>

        <Card title="Learning curve" sub={`${MODEL_LABEL[lcModel]}: R² vs number of training rows${ds.splits.length > 1 ? ` · always on the ${ds.cv[ds.main_split]!.label.toLowerCase()} split` : ""}`}>
          <Legend items={[{ label: "Training R²", color: MODEL_COLOR[lcModel], kind: "dash" }, { label: "Held-out R²", color: MODEL_COLOR[lcModel] }]} />
          <Plot height={300} xDomain={[0, lc.sizes[lc.sizes.length - 1]]} yDomain={[Math.min(0, ...lcd.test_r2), 1]} xLabel="Training rows" yLabel="R²" ariaLabel="Learning curve" zeroLine>
            {(ctx) => {
              const { x, y } = ctx;
              const mk = (v: number[]) => line<number>().x((_, i) => x(lc.sizes[i])).y((v) => y(v))(v) ?? "";
              return (
                <g>
                  <path d={mk(lcd.train_r2)} fill="none" stroke={MODEL_COLOR[lcModel]} strokeWidth={2} strokeDasharray="6 4" />
                  <path d={mk(lcd.test_r2)} fill="none" stroke={MODEL_COLOR[lcModel]} strokeWidth={2.8} />
                  {lc.sizes.map((s, i) => <circle key={i} cx={x(s)} cy={y(lcd.test_r2[i])} r={4} fill={MODEL_COLOR[lcModel]} />)}
                  <HoverColumns xs={lc.sizes} ctx={ctx} render={(i) => (
                    <>
                      <TipRow label={<b>{lc.sizes[i]} rows</b>} value="" />
                      <TipRow label="Training R²" value={fmt(lcd.train_r2[i], 3)} />
                      <TipRow label="Held-out R²" value={fmt(lcd.test_r2[i], 3)} />
                    </>
                  )} />
                </g>
              );
            }}
          </Plot>
          <p className="small ink2">
            {finalGap > 0.15
              ? <>A wide gap at full size ({fmt(finalGap, 2)}): overfitting. Simplify or regularize, or get more data if the held-out curve is still rising.</>
              : <>Training and held-out curves meet (gap {fmt(finalGap, 2)}). If they're flat, more data of the same kind won't help much; you need better predictors.</>}
          </p>
        </Card>

        <Card title={`Subgroups: ${ds.subgroup_name}`} sub="Held-out RMSE by group. Same model, different error.">
          <TrainTestBars showTrain={false} valueLabel={`RMSE (${ds.units})`} digits={1} rowHeight={Math.max(30, Math.min(44, 220 / d.sub.length))} labelWidth={150}
            domain={[0, Math.max(...d.sub.map((s) => s.rmse)) * 1.1]}
            rows={d.sub.map((s) => ({ id: s.g, label: `${s.g} (n=${s.n})`, color, test: s.rmse }))} />
          <p className="small ink2">
            Check every group the model will serve. A higher RMSE in a small group can be noise, so look at n, and at whether the gap persists across folds and seeds.
          </p>
        </Card>
      </div>
      <Callout>
        <b>Stay inside your data.</b> {ds.name}: the outcome ranges from {fmt(ds.y_summary.min, 0)} to {fmt(ds.y_summary.max, 0)} {ds.units}. Trees can't extrapolate at all; lines
        extrapolate confidently and wrongly. Neither is safe outside the data.
      </Callout>
      <SourceNote>{ds.source}. Split: {d.cv.description}.</SourceNote>
    </div>
  );
}

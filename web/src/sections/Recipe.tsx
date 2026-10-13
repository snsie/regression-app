import { useEffect, useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";
import { useWidth } from "../charts/kit";
import { Card, Callout, DatasetPicker, Loading, Slider, SourceNote, Toggle } from "../components/ui";
import { DatasetId, fmt, fmtSigned, useDataset } from "../data";

const IDS: DatasetId[] = ["nhanes_glucose", "nhanes_sbp", "diabetes"];

export default function Recipe() {
  const [id, setId] = useState<DatasetId>("nhanes_glucose");
  const ds = useDataset(id);
  const [vals, setVals] = useState<Record<string, number>>({});
  const [ref, width] = useWidth<HTMLDivElement>();

  useEffect(() => {
    if (ds) setVals(Object.fromEntries(ds.recipe.features.map((f) => [f.id, f.binary ? 0 : f.median])));
  }, [ds]);

  const contrib = useMemo(() => {
    if (!ds) return [];
    return ds.recipe.features.map((f) => ({ ...f, value: vals[f.id] ?? f.median, delta: f.weight * ((vals[f.id] ?? f.median) - f.mean) }));
  }, [ds, vals]);

  if (!ds || Object.keys(vals).length === 0) return <Loading />;
  const r = ds.recipe;
  const avgPred = r.intercept + r.features.reduce((s, f) => s + f.weight * f.mean, 0);
  const pred = r.intercept + contrib.reduce((s, f) => s + f.weight * f.value, 0);
  const cv = ds.cv[ds.main_split]!.models.linear;

  // waterfall geometry
  const steps: { label: string; from: number; to: number; delta?: number }[] = [];
  let run = avgPred;
  for (const c of contrib) {
    steps.push({ label: c.label, from: run, to: run + c.delta, delta: c.delta });
    run += c.delta;
  }
  const all = [avgPred, pred, ...steps.flatMap((s) => [s.from, s.to])];
  const lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.15 || 5;
  const lw = width < 560 ? 130 : 210;
  const x = scaleLinear().domain([lo - pad, hi + pad]).range([lw, width - 60]);
  const rowH = 30;
  const rows = steps.length + 2;
  const H = rows * rowH + 24;

  return (
    <div className="stack">
      <div className="controls-bar">
        <DatasetPicker value={id} onChange={setId} ids={IDS} />
      </div>
      <Card>
        <div className="stat-label">The fitted recipe (all {ds.n_rows.toLocaleString()} rows, original units)</div>
        <div className="tnum" style={{ fontSize: "1.05rem", lineHeight: 1.8, marginTop: 6 }}>
          <b>{ds.target_label}</b> = <b>{fmt(r.intercept, 1)}</b>
          {r.features.map((f) => (
            <span key={f.id}>
              {" "}{f.weight >= 0 ? "+" : "−"} <b>{fmt(Math.abs(f.weight), Math.abs(f.weight) < 1 ? 3 : 2)}</b> × {f.label}
            </span>
          ))}
        </div>
      </Card>

      <div className="split-wide">
        <Card title="Build one prediction" sub="Each bar is one predictor's weight × how far this patient is from the average patient">
          <div ref={ref}>
            <svg width={width} height={H} role="img" aria-label="Waterfall of each predictor's contribution to the prediction">
              {[0, rows - 1].map((ri) => {
                const v = ri === 0 ? avgPred : pred;
                return (
                  <g key={ri}>
                    <text x={lw - 10} y={ri * rowH + rowH / 2} dy="0.32em" textAnchor="end" fontSize={13} fontWeight={650} fill="var(--ink)">
                      {ri === 0 ? "Average patient" : "This patient"}
                    </text>
                    <rect x={Math.min(x(lo - pad), x(v))} y={ri * rowH + 6} width={Math.abs(x(v) - x(lo - pad))} height={rowH - 12} rx={4}
                      fill={ri === 0 ? "var(--m-baseline)" : "var(--m-linear)"} />
                    <text x={x(v) + 6} y={ri * rowH + rowH / 2} dy="0.32em" fontSize={13} fontWeight={700} fill="var(--ink)" className="tnum">{fmt(v, 1)}</text>
                  </g>
                );
              })}
              {steps.map((s, i) => {
                const yy = (i + 1) * rowH;
                const up = (s.delta ?? 0) >= 0;
                return (
                  <g key={s.label}>
                    <text x={lw - 10} y={yy + rowH / 2} dy="0.32em" textAnchor="end" fontSize={12.5} fill="var(--ink-2)">{s.label}</text>
                    <line x1={x(s.from)} x2={x(s.from)} y1={yy - 2} y2={yy + 6} stroke="var(--axis)" />
                    <rect x={Math.min(x(s.from), x(s.to))} y={yy + 7} width={Math.max(1.5, Math.abs(x(s.to) - x(s.from)))} height={rowH - 14} rx={3}
                      fill={up ? "var(--resid-pos)" : "var(--resid-neg)"} opacity={0.85} />
                    <text x={Math.max(x(s.from), x(s.to)) + 6} y={yy + rowH / 2} dy="0.32em" fontSize={12} fill="var(--ink-2)" className="tnum">
                      {fmtSigned(s.delta ?? 0, 1)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
          <p className="small muted">Red adds to the prediction, blue subtracts. Every patient gets the same weight per unit: that's what “linear” means.</p>
        </Card>

        <Card title="Patient" sub="Move the sliders; the arithmetic updates live">
          <div className="stack" style={{ gap: 6 }}>
            {r.features.map((f) =>
              f.binary ? (
                <Toggle key={f.id} checked={(vals[f.id] ?? 0) === 1} onChange={(c) => setVals({ ...vals, [f.id]: c ? 1 : 0 })}>
                  {f.label}
                </Toggle>
              ) : (
                <Slider key={f.id} label={f.label} value={vals[f.id]} min={f.min} max={f.max} step={(f.max - f.min) / 100}
                  onChange={(v) => setVals({ ...vals, [f.id]: v })} format={(v) => fmt(v, f.max - f.min > 20 ? 0 : 1)} />
              ),
            )}
          </div>
          <div className="spacer" />
          <div className="stat-label">Prediction</div>
          <div className="stat-value tnum">{fmt(pred, 1)} <span style={{ fontSize: "1rem" }} className="muted">{ds.units}</span></div>
          <div className="small ink2">
            Typical miss on new patients: ±{fmt(cv.mean_test_rmse, 1)} {ds.units} (cross-validated RMSE). Held-out R² = {fmt(cv.mean_test_r2, 2)}.
          </div>
        </Card>
      </div>

      {id === "nhanes_glucose" && (
        <Callout>
          <b>In the clinic: HbA1c → average glucose.</b> The ADAG study (507 people, ~2,700 glucose readings each over 3 months) fit
          eAG = 28.7 × A1C − 46.7 with R² = 0.84. That formula is how labs report an “estimated average glucose”. Here the outcome is a single
          <i> fasting</i> glucose, so expect a similar slope with more scatter.
          <div className="small muted" style={{ marginTop: 6 }}>Nathan DM et al. Diabetes Care 2008;31:1473–8. doi:10.2337/dc08-0545</div>
        </Callout>
      )}
      <div className="grid-2">
        <Callout tone="good"><b>Reach for it when</b> the relationship is close to straight, data are small, or people must be able to read and audit the model.</Callout>
        <Callout tone="warn"><b>Watch out for</b> curves and interactions it can't see unless you add them, and single extreme patients pulling the line.</Callout>
      </div>
      <SourceNote>{ds.source}. Weights are from an unscaled least-squares fit on all rows; the ± error is from 5-fold cross-validation.</SourceNote>
    </div>
  );
}

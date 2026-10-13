import { useEffect, useState } from "react";
import { line } from "d3-shape";
import { HoverColumns, Legend, Plot, TipRow, extent } from "../charts/kit";
import { Card, Callout, DatasetPicker, Loading, ModelPicker, SourceNote } from "../components/ui";
import { DATASET_IDS, DATASET_SHORT, DatasetData, DatasetId, fmt, load, mean, MODEL_COLOR, MODEL_IDS, MODEL_LABEL, ModelId, sd, SplitId } from "../data";

const ROWS: { ds: DatasetId; split: SplitId; label: string }[] = [
  { ds: "nhanes_glucose", split: "kfold", label: "HbA1c → glucose" },
  { ds: "nhanes_sbp", split: "kfold", label: "Blood pressure" },
  { ds: "diabetes", split: "kfold", label: "Diabetes (n=442)" },
  { ds: "parkinsons", split: "random", label: "Parkinson's · random rows" },
  { ds: "parkinsons", split: "temporal", label: "Parkinson's · known patients" },
  { ds: "parkinsons", split: "group", label: "Parkinson's · new patients" },
];

function cellColor(v: number) {
  if (Math.abs(v) < 0.03) return "var(--div-mid)";
  if (v < 0) return `color-mix(in srgb, var(--div-pos) ${Math.round(Math.min(1, -v / 0.5) * 70) + 15}%, var(--div-mid))`;
  return `color-mix(in srgb, var(--div-neg) ${Math.round(Math.min(1, v) * 80) + 5}%, var(--div-mid))`;
}

export default function Compare() {
  const [all, setAll] = useState<Record<DatasetId, DatasetData> | null>(null);
  const [foldDs, setFoldDs] = useState<DatasetId>("diabetes");
  const [a, setA] = useState<ModelId>("linear");
  const [b, setB] = useState<ModelId>("rf");
  useEffect(() => {
    Promise.all(DATASET_IDS.map((id) => load<DatasetData>(id))).then((arr) => setAll(Object.fromEntries(arr.map((d) => [d.id, d])) as Record<DatasetId, DatasetData>));
  }, []);
  if (!all) return <Loading />;

  const fds = all[foldDs];
  const fcv = fds.cv[fds.main_split]!;
  const pa = fcv.models[a].repeated_r2, pb = fcv.models[b].repeated_r2;
  const wins = pa.filter((v, i) => v > pb[i]).length;
  const pdom = extent([...pa, ...pb], 0.06);
  const reductions = MODEL_IDS.filter((m) => m !== "baseline").map((m) => {
    const r = fcv.models[m].repeated_rmse.map((v, i) => 1 - v / fcv.models.baseline.repeated_rmse[i]);
    return { m, mean: mean(r), sd: sd(r) };
  });

  return (
    <div className="stack">
      <Card title="Cross-validated R² on held-out data" sub="Mean ± SD over 10 repeats of 5-fold CV (temporal split: 5 folds). 0 = predicting the average. ★ = best in row.">
        <div className="table-scroll">
          <table className="data" style={{ minWidth: 720 }}>
            <thead>
              <tr>
                <th>Dataset · split</th>
                {MODEL_IDS.map((m) => <th key={m} style={{ textAlign: "center" }}><span className="swatch" style={{ background: MODEL_COLOR[m], marginRight: 6 }} />{MODEL_LABEL[m]}</th>)}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => {
                const cv = all[r.ds].cv[r.split]!;
                const vals = MODEL_IDS.map((m) => ({ m, mean: mean(cv.models[m].repeated_r2), sd: sd(cv.models[m].repeated_r2) }));
                const best = vals.reduce((x, y) => (y.mean > x.mean ? y : x));
                return (
                  <tr key={r.label}>
                    <td style={{ fontWeight: 600 }}>{r.label}</td>
                    {vals.map((v) => (
                      <td key={v.m} style={{ textAlign: "center", background: cellColor(v.mean), color: Math.abs(v.mean) > 0.55 ? "#fff" : undefined, fontWeight: v === best ? 750 : 450 }}
                        title={`${MODEL_LABEL[v.m]}: ${fmt(v.mean, 3)} ± ${fmt(v.sd, 3)}`}>
                        {v === best && "★ "}{fmt(v.mean, 2)}<span style={{ opacity: 0.75, fontSize: "0.8em" }}> ±{fmt(v.sd, 2)}</span>
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="row small muted" style={{ marginTop: 10, gap: 8 }}>
          <span>−0.5</span>
          <span style={{ width: 160, height: 10, borderRadius: 4, background: "linear-gradient(90deg, var(--div-pos), var(--div-mid), var(--div-neg))" }} />
          <span>1.0 R²</span>
        </div>
      </Card>

      <div className="grid-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
        <Card title="Nearly linear"><div className="small ink2">HbA1c → glucose: every model lands within a few hundredths of the line, and only the line fits on an index card.</div></Card>
        <Card title="Small n"><div className="small ink2">Diabetes, 442 patients: linear beats both tree ensembles (default boosting overfits). The foundation model ties it, within the noise.</div></Card>
        <Card title="Rich signal"><div className="small ink2">Known Parkinson's patients: every flexible model leaves linear far behind.</div></Card>
        <Card title="No signal"><div className="small ink2">New Parkinson's patients: every model is worse than predicting the average.</div></Card>
      </div>

      <div className="controls-bar">
        <DatasetPicker value={foldDs} onChange={setFoldDs} ids={DATASET_IDS} label="Dataset for the plots below" />
      </div>
      <div className="grid-2">
        <Card title="Do the lines agree?" sub={`Held-out R² in each of the 5 folds · ${DATASET_SHORT[foldDs]} · ${fcv.label}`}>
          <Legend items={MODEL_IDS.map((m) => ({ label: MODEL_LABEL[m], color: MODEL_COLOR[m] }))} />
          {(() => {
            const vals = MODEL_IDS.flatMap((m) => fcv.models[m].folds.map((f) => f.test_r2));
            return (
              <Plot height={300} xDomain={[0.8, 5.2]} yDomain={extent(vals, 0.1)} xTicks={[1, 2, 3, 4, 5]} xFormat={(v) => `Fold ${v}`} yLabel="Held-out R²" ariaLabel="Per-fold R² by model" nice={false} zeroLine>
                {(ctx) => (
                  <g>
                    {MODEL_IDS.map((m) => {
                      const v = fcv.models[m].folds.map((f) => f.test_r2);
                      return (
                        <g key={m}>
                          <path d={line<number>().x((_, i) => ctx.x(i + 1)).y((d) => ctx.y(d))(v) ?? ""} fill="none" stroke={MODEL_COLOR[m]} strokeWidth={2.2} strokeDasharray={m === "baseline" ? "5 4" : undefined} />
                          {v.map((d, i) => <circle key={i} cx={ctx.x(i + 1)} cy={ctx.y(d)} r={4} fill={MODEL_COLOR[m]} stroke="var(--surface)" strokeWidth={1.5} />)}
                        </g>
                      );
                    })}
                    <HoverColumns xs={[1, 2, 3, 4, 5]} ctx={ctx} render={(i) => (
                      <>
                        <TipRow label={<b>Fold {i + 1}</b>} value="" />
                        {MODEL_IDS.map((m) => <TipRow key={m} color={MODEL_COLOR[m]} label={MODEL_LABEL[m]} value={fmt(fcv.models[m].folds[i].test_r2, 3)} />)}
                      </>
                    )} />
                  </g>
                )}
              </Plot>
            );
          })()}
          <p className="small ink2">Some folds are hard for every model. That's why one train/test split can mislead: report mean ± SD across folds.</p>
        </Card>

        <Card title="Is the gap bigger than the spread?" sub={`Paired: the same ${pa.length} train/test splits for both models`}>
          <div className="row">
            <ModelPicker value={a} onChange={setA} models={MODEL_IDS.filter((m) => m !== "baseline")} label="Model A (x-axis)" />
            <ModelPicker value={b} onChange={setB} models={MODEL_IDS.filter((m) => m !== "baseline")} label="Model B (y-axis)" />
          </div>
          <Plot height={270} xDomain={pdom} yDomain={pdom} xLabel={`${MODEL_LABEL[a]} R²`} yLabel={`${MODEL_LABEL[b]} R²`} ariaLabel="Paired comparison of two models across repeated splits" nice={false}>
            {({ x, y, show, hide }) => (
              <g>
                <path d={`M${x(pdom[0])},${y(pdom[0])} L${x(pdom[1])},${y(pdom[0])} L${x(pdom[1])},${y(pdom[1])} Z`} fill={MODEL_COLOR[a]} opacity={0.07} />
                <path d={`M${x(pdom[0])},${y(pdom[0])} L${x(pdom[0])},${y(pdom[1])} L${x(pdom[1])},${y(pdom[1])} Z`} fill={MODEL_COLOR[b]} opacity={0.07} />
                <line x1={x(pdom[0])} y1={y(pdom[0])} x2={x(pdom[1])} y2={y(pdom[1])} stroke="var(--ink-2)" strokeDasharray="5 4" />
                <text x={x(pdom[1]) - 6} y={y(pdom[0]) - 8} textAnchor="end" fontSize={12} fill="var(--ink-2)">{MODEL_LABEL[a]} better</text>
                <text x={x(pdom[0]) + 6} y={y(pdom[1]) + 16} fontSize={12} fill="var(--ink-2)">{MODEL_LABEL[b]} better</text>
                {pa.map((v, i) => (
                  <circle key={i} cx={x(v)} cy={y(pb[i])} r={4.5} fill={v > pb[i] ? MODEL_COLOR[a] : MODEL_COLOR[b]} stroke="var(--surface)" strokeWidth={1.2} opacity={0.85}
                    onMouseEnter={() => show(x(v), y(pb[i]), <><TipRow label={MODEL_LABEL[a]} value={fmt(v, 3)} /><TipRow label={MODEL_LABEL[b]} value={fmt(pb[i], 3)} /></>)} onMouseLeave={hide} />
                ))}
              </g>
            )}
          </Plot>
          <p className="small ink2">
            {a === b ? "Pick two different models." : <><b>{MODEL_LABEL[a]}</b> beats {MODEL_LABEL[b]} on <b>{wins} of {pa.length}</b> splits.
              {wins / pa.length > 0.8 || wins / pa.length < 0.2 ? " A consistent winner." : " Too close to call: the gap is within the noise."}</>}
          </p>
        </Card>
      </div>

      <Card title="How much better than nothing?" sub={`Reduction in held-out RMSE vs predicting the mean, ${DATASET_SHORT[foldDs]} (mean ± SD over repeated splits)`}>
        <div className="grid-4">
          {reductions.map((r) => (
            <div key={r.m}>
              <div className="row small" style={{ gap: 6 }}><span className="swatch" style={{ background: MODEL_COLOR[r.m] }} />{MODEL_LABEL[r.m]}</div>
              <div className="stat-value tnum" style={{ color: r.mean < 0 ? "var(--critical)" : undefined }}>{fmt(r.mean * 100, 0)}%</div>
              <div className="small muted">± {fmt(r.sd * 100, 0)} points</div>
            </div>
          ))}
        </div>
      </Card>
      <Callout>Fair comparison = <b>same folds</b> + <b>many splits</b> + <b>a baseline</b>. Then a “win” means something.</Callout>
      <SourceNote>Every model in every row was scored on identical folds with random_state fixed, and preprocessing ran inside a pipeline.</SourceNote>
    </div>
  );
}

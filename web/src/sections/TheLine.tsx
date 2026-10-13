import { useEffect, useMemo, useState } from "react";
import { Plot, TipRow, Legend } from "../charts/kit";
import { Card, Callout, Loading, PauseThink, Segmented, Slider, Toggle } from "../components/ui";
import { fmt, fmtSigned, leastAbsolute, leastSquares, mean, PlaygroundData, useData } from "../data";
import { PAIR_OPTIONS, PairId } from "./Framing";

const EXAMPLE_X: Record<PairId, number> = { age_sbp: 55, bmi_sbp: 28, a1c_glucose: 7 };
const DIFF_X: Record<PairId, number> = { age_sbp: 10, bmi_sbp: 5, a1c_glucose: 1 };

export default function TheLine() {
  const pg = useData<PlaygroundData>("playground");
  const [pair, setPair] = useState<PairId>("age_sbp");
  const [loss, setLoss] = useState<"sq" | "abs">("sq");
  const [showSquares, setShowSquares] = useState(false);
  const [outlier, setOutlier] = useState(false);
  const [outY, setOutY] = useState(0);
  const p = pg?.pairs[pair];

  const base = useMemo(() => {
    if (!p) return null;
    const xbar = mean(p.x);
    const ys = [...p.y].sort((a, b) => a - b);
    return { xbar, ymin: ys[0], ymax: ys[ys.length - 1], xmax: Math.max(...p.x), xmin: Math.min(...p.x) };
  }, [p]);

  // the line is parameterized as (height at x̄, slope) so sliders stay intuitive
  const [level, setLevel] = useState(0);
  const [slope, setSlope] = useState(0);

  const data = useMemo(() => {
    if (!p || !base) return null;
    if (!outlier) return { x: p.x, y: p.y };
    return { x: [...p.x, base.xmax], y: [...p.y, outY] };
  }, [p, base, outlier, outY]);

  const ls = useMemo(() => (data ? leastSquares(data.x, data.y) : null), [data]);
  const lad = useMemo(() => (data ? leastAbsolute(data.x, data.y) : null), [data]);

  useEffect(() => {
    if (!p || !base) return;
    // start each dataset with a flat line at the mean: the baseline
    setLevel(mean(p.y));
    setSlope(0);
    setOutY(base.ymin + (base.ymax - base.ymin) * 0.02);
    setOutlier(false);
  }, [pair, p, base]);

  if (!pg || !p || !base || !data || !ls || !lad) return <Loading />;

  const intercept = level - slope * base.xbar;
  const pred = (v: number) => intercept + slope * v;
  const res = data.x.map((v, i) => data.y[i] - pred(v));
  const sse = res.reduce((s, e) => s + e * e, 0);
  const sae = res.reduce((s, e) => s + Math.abs(e), 0);
  const minSse = data.x.reduce((s, v, i) => s + (data.y[i] - (ls.intercept + ls.slope * v)) ** 2, 0);
  const minSae = data.x.reduce((s, v, i) => s + Math.abs(data.y[i] - (lad.intercept + lad.slope * v)), 0);
  const cur = loss === "sq" ? sse : sae;
  const best = loss === "sq" ? minSse : minSae;
  const ratio = cur / best;

  const lsBase = leastSquares(p.x, p.y);
  const slopeRange = Math.abs(lsBase.slope) * 3 + 1e-6;
  const ySpan = base.ymax - base.ymin;
  const xd: [number, number] = [base.xmin - (base.xmax - base.xmin) * 0.04, base.xmax + (base.xmax - base.xmin) * 0.04];
  const yd: [number, number] = [Math.min(base.ymin, outY) - ySpan * 0.04, base.ymax + ySpan * 0.04];
  const ex = EXAMPLE_X[pair];
  const unit1 = p.x_units.replace(/s$/, "");
  const snap = (f: { slope: number; intercept: number }) => {
    setSlope(f.slope);
    setLevel(f.intercept + f.slope * base.xbar);
  };

  return (
    <div className="stack">
      <div className="split-wide">
        <Card>
          <div className="controls-bar">
            <Segmented label="Data" value={pair} onChange={setPair} options={PAIR_OPTIONS} />
            <Segmented label="Loss" value={loss} onChange={setLoss} options={[{ value: "sq", label: "Squared error" }, { value: "abs", label: "Absolute error" }]} />
          </div>
          <div className="controls-bar">
            <Slider label={`Slope (${p.y_units} per ${unit1})`} value={slope} min={-slopeRange} max={slopeRange} step={slopeRange / 200}
              onChange={setSlope} format={(v) => fmtSigned(v, 2)} />
            <Slider label={`Height at average ${p.x_label} (${fmt(base.xbar, 1)})`} value={level} min={base.ymin} max={base.ymax}
              step={(base.ymax - base.ymin) / 300} onChange={setLevel} format={(v) => fmt(v, 1)} />
          </div>
          <div className="row" style={{ marginBottom: 8 }}>
            <button className="btn primary" onClick={() => snap(ls)}>Snap to least squares</button>
            <button className="btn" onClick={() => snap(lad)}>Snap to least absolute error</button>
            <button className="btn" onClick={() => { setSlope(0); setLevel(mean(data.y)); }}>Flat line (baseline)</button>
          </div>
          <Legend items={[
            { label: "Your line", color: "var(--ink)" },
            { label: "Residual: underpredicted (e > 0)", color: "var(--resid-pos)" },
            { label: "Overpredicted (e < 0)", color: "var(--resid-neg)" },
            ...(outlier ? [{ label: "Least squares / least absolute", color: "var(--m-linear)", kind: "dash" as const }] : []),
          ]} />
          <Plot height={400} xDomain={xd} yDomain={yd} nice={false}
            xLabel={`${p.x_label} (${p.x_units})`} yLabel={`${p.y_label} (${p.y_units})`}
            ariaLabel="Scatter plot with an adjustable line and residuals">
            {({ x, y, show, hide }) => (
              <g>
                {showSquares && data.x.map((xv, i) => {
                  const s = Math.abs(y(data.y[i]) - y(pred(xv)));
                  return <rect key={`s${i}`} x={x(xv)} y={Math.min(y(data.y[i]), y(pred(xv)))} width={s} height={s}
                    fill={res[i] > 0 ? "var(--resid-pos)" : "var(--resid-neg)"} opacity={0.06} />;
                })}
                {data.x.map((xv, i) => (
                  <line key={`r${i}`} x1={x(xv)} x2={x(xv)} y1={y(data.y[i])} y2={y(pred(xv))}
                    stroke={res[i] > 0 ? "var(--resid-pos)" : "var(--resid-neg)"} strokeWidth={1} strokeDasharray="3 2" opacity={0.6} />
                ))}
                {data.x.map((xv, i) => {
                  const isOut = outlier && i === data.x.length - 1;
                  return (
                    <circle key={i} cx={x(xv)} cy={y(data.y[i])} r={isOut ? 8 : 4}
                      fill={isOut ? "var(--critical)" : "var(--muted)"} fillOpacity={isOut ? 1 : 0.55} stroke="var(--surface)" strokeWidth={isOut ? 2 : 1}
                      onMouseEnter={() => show(x(xv), y(data.y[i]), (
                        <>
                          {isOut && <b>Added extreme patient</b>}
                          <TipRow label={p.x_label} value={fmt(xv, 1)} />
                          <TipRow label="Observed" value={fmt(data.y[i], 0)} />
                          <TipRow label="Predicted" value={fmt(pred(xv), 1)} />
                          <TipRow label="Residual e" value={fmtSigned(res[i], 1)} />
                        </>
                      ))}
                      onMouseLeave={hide} />
                  );
                })}
                {outlier && (
                  <>
                    <line x1={x(xd[0])} x2={x(xd[1])} y1={y(ls.intercept + ls.slope * xd[0])} y2={y(ls.intercept + ls.slope * xd[1])}
                      stroke="var(--m-linear)" strokeWidth={2} strokeDasharray="7 4" />
                    <text x={x(xd[1]) - 4} y={y(ls.intercept + ls.slope * xd[1]) - 8} textAnchor="end" fontSize={12} fill="var(--ink-2)">least squares</text>
                    <line x1={x(xd[0])} x2={x(xd[1])} y1={y(lad.intercept + lad.slope * xd[0])} y2={y(lad.intercept + lad.slope * xd[1])}
                      stroke="var(--m-linear)" strokeWidth={2} strokeDasharray="2 3" />
                    <text x={x(xd[1]) - 4} y={y(lad.intercept + lad.slope * xd[1]) + 16} textAnchor="end" fontSize={12} fill="var(--ink-2)">least absolute</text>
                  </>
                )}
                <line x1={x(xd[0])} x2={x(xd[1])} y1={y(pred(xd[0]))} y2={y(pred(xd[1]))} stroke="var(--ink)" strokeWidth={2.5} />
              </g>
            )}
          </Plot>
          <div className="row" style={{ marginTop: 8 }}>
            <Toggle checked={showSquares} onChange={setShowSquares}>Draw each squared residual</Toggle>
            <Toggle checked={outlier} onChange={setOutlier}>Add one extreme patient</Toggle>
            {outlier && (
              <Slider label="Extreme patient's value" value={outY} min={base.ymin - ySpan * 0.6} max={base.ymax + ySpan * 0.6} step={ySpan / 200}
                onChange={setOutY} format={(v) => `${fmt(v, 0)} ${p.y_units}`} width={300} />
            )}
          </div>
        </Card>

        <div className="stack">
          <Card>
            <div className="stat-label">Your line</div>
            <div style={{ fontSize: "1.3rem", fontWeight: 650, margin: "6px 0" }} className="tnum">
              ŷ = {fmt(intercept, 1)} {slope >= 0 ? "+" : "−"} {fmt(Math.abs(slope), 2)} × {p.x_label}
            </div>
            <div className="stat-label" style={{ marginTop: 14 }}>{loss === "sq" ? "Sum of squared residuals Σe²" : "Sum of absolute residuals Σ|e|"}</div>
            <div className="stat-value tnum">{fmt(cur, 0)}</div>
            <div style={{ height: 10, background: "var(--grid)", borderRadius: 6, overflow: "hidden" }} aria-hidden>
              <div style={{ height: "100%", width: `${Math.min(100, 100 / ratio)}%`, background: ratio < 1.001 ? "var(--good)" : "var(--accent)", transition: "width .2s" }} />
            </div>
            <div className="small ink2" style={{ marginTop: 6 }}>
              {ratio < 1.001 ? <>✓ This is the minimum: the {loss === "sq" ? "least-squares" : "least-absolute-error"} line.</> : <>{fmt((ratio - 1) * 100, 0)}% above the smallest possible value ({fmt(best, 0)}).</>}
            </div>
          </Card>
          <Card title="Read it in clinical units">
            <div className="small ink2">
              <p><b>Slope</b> {fmtSigned(slope, 2)} {p.y_units} per 1 {unit1} higher {p.x_label}. It compares patients; it doesn't prove cause.</p>
              <p><b>Intercept</b> {fmt(intercept, 1)} {p.y_units} at {p.x_label} = 0. The data cover {fmt(base.xmin, 1)}–{fmt(base.xmax, 1)} {p.x_units}, so it's an anchor, not a clinical estimate.</p>
              <p style={{ marginBottom: 0 }}><b>Worked example</b>: {p.x_label} {ex} → {fmt(intercept, 1)} {slope >= 0 ? "+" : "−"} {fmt(Math.abs(slope), 2)} × {ex} ≈ <b>{fmt(pred(ex), 0)} {p.y_units}</b></p>
            </div>
          </Card>
          <PauseThink prompt={`Two patients differ by ${DIFF_X[pair]} ${p.x_units} of ${p.x_label}. How far apart are their predictions?`}>
            <div className="small">{DIFF_X[pair]} × slope = {DIFF_X[pair]} × {fmt(slope, 2)} = <b>{fmt(DIFF_X[pair] * slope, 1)} {p.y_units}</b>. With a straight line the gap is the same anywhere on the scale.</div>
          </PauseThink>
        </div>
      </div>
      <Callout>
        <b>Why square?</b> Misses can't cancel, big misses cost more, and there is one exact answer. <b>The cost:</b> turn on the extreme patient and drag it.
        One patient pulls the least-squares line much further than the least-absolute-error line. A big residual is a patient worth a closer look, not deletion.
        In code, <code>scipy.stats.linregress(x, y)</code> or <code>LinearRegression().fit(X, y)</code> does the fit in one call.
      </Callout>
    </div>
  );
}

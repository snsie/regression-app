import { useMemo, useState } from "react";
import { Plot, TipRow, extent, Legend } from "../charts/kit";
import { Card, Callout, Loading, Segmented, SourceNote, Toggle } from "../components/ui";
import { fmt, leastSquares, PlaygroundData, r2, useData } from "../data";

export type PairId = "age_sbp" | "bmi_sbp" | "a1c_glucose";
export const PAIR_OPTIONS: { value: PairId; label: string }[] = [
  { value: "age_sbp", label: "Age → SBP" },
  { value: "bmi_sbp", label: "BMI → SBP" },
  { value: "a1c_glucose", label: "HbA1c → glucose" },
];

export default function Framing() {
  const pg = useData<PlaygroundData>("playground");
  const [pair, setPair] = useState<PairId>("age_sbp");
  const [showF, setShowF] = useState(true);
  const [showE, setShowE] = useState(false);
  const p = pg?.pairs[pair];
  const fit = useMemo(() => (p ? leastSquares(p.x, p.y) : null), [p]);
  if (!pg || !p || !fit) return <Loading />;
  const pred = p.x.map((v) => fit.intercept + fit.slope * v);
  const R2 = r2(p.y, pred);
  const xd = extent(p.x, 0.04);
  const yd = extent(p.y, 0.04);

  return (
    <div className="stack">
      <Card>
        <div className="eq">
          y = <span className="t-f">f(x)</span> + <span className="t-e">ε</span>
        </div>
        <div className="grid-3 small ink2" style={{ textAlign: "center" }}>
          <div><b>y</b>: a number to predict (blood pressure, glucose, a symptom score)</div>
          <div><b style={{ color: "var(--m-linear)" }}>f(x)</b>: the pattern we can learn from predictors x</div>
          <div><b style={{ color: "var(--critical)" }}>ε</b>: everything we can't: measurement noise, unmeasured factors</div>
        </div>
      </Card>

      <div className="split-wide">
        <Card title={`${p.y_label} vs ${p.x_label}`} sub={`300 randomly sampled NHANES 2017–2018 adults (of ${p.population_n.toLocaleString()} with both measured)`}>
          <div className="controls-bar">
            <Segmented label="Predictor → outcome" value={pair} onChange={setPair} options={PAIR_OPTIONS} />
            <Toggle checked={showF} onChange={setShowF}>Show f (best straight line)</Toggle>
            <Toggle checked={showE} onChange={setShowE}>Show ε (what's left over)</Toggle>
          </div>
          <Legend items={[
            { label: "Patients", color: "var(--muted)", kind: "dot" },
            ...(showF ? [{ label: "f(x): least-squares line", color: "var(--m-linear)" }] : []),
            ...(showE ? [{ label: "ε: residual above / below", color: "var(--critical)" }] : []),
          ]} />
          <Plot
            height={380} xDomain={xd} yDomain={yd}
            xLabel={`${p.x_label} (${p.x_units})`} yLabel={`${p.y_label} (${p.y_units})`}
            ariaLabel={`Scatter plot of ${p.y_label} against ${p.x_label}`}
          >
            {({ x, y, show, hide }) => (
              <g>
                {showE && p.x.map((xv, i) => (
                  <line key={`e${i}`} x1={x(xv)} x2={x(xv)} y1={y(p.y[i])} y2={y(pred[i])}
                    stroke={p.y[i] > pred[i] ? "var(--resid-pos)" : "var(--resid-neg)"} strokeWidth={1} opacity={0.55} />
                ))}
                {p.x.map((xv, i) => (
                  <circle key={i} cx={x(xv)} cy={y(p.y[i])} r={4} fill="var(--muted)" fillOpacity={0.5} stroke="var(--surface)" strokeWidth={1}
                    onMouseEnter={() => show(x(xv), y(p.y[i]), (
                      <>
                        <TipRow label={p.x_label} value={`${fmt(xv, 1)} ${p.x_units}`} />
                        <TipRow label={p.y_label} value={`${fmt(p.y[i], 0)} ${p.y_units}`} />
                        <TipRow label="ε (residual)" value={fmt(p.y[i] - pred[i], 1)} />
                      </>
                    ))}
                    onMouseLeave={hide} />
                ))}
                {showF && (
                  <line x1={x(xd[0])} x2={x(xd[1])} y1={y(fit.intercept + fit.slope * xd[0])} y2={y(fit.intercept + fit.slope * xd[1])}
                    stroke="var(--m-linear)" strokeWidth={2.5} />
                )}
              </g>
            )}
          </Plot>
        </Card>

        <div className="stack">
          <Card title="How much does f explain?" sub="Share of the variation in y, on these 300 patients">
            <div style={{ display: "flex", height: 34, borderRadius: 6, overflow: "hidden", gap: 2, margin: "8px 0" }} role="img"
              aria-label={`f explains ${fmt(R2 * 100, 0)} percent, epsilon ${fmt(100 - R2 * 100, 0)} percent`}>
              <div style={{ width: `${Math.max(0.5, R2 * 100)}%`, background: "var(--m-linear)", transition: "width .4s" }} />
              <div style={{ flex: 1, background: "color-mix(in srgb, var(--critical) 55%, var(--surface))" }} />
            </div>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span><b style={{ color: "var(--m-linear)" }}>f(x)</b> {fmt(R2 * 100, 0)}%</span>
              <span><b style={{ color: "var(--critical)" }}>ε</b> {fmt(100 - R2 * 100, 0)}%</span>
            </div>
            <p className="small ink2" style={{ marginTop: 10 }}>
              This share is R². With one straight line: R² = <b className="tnum">{fmt(R2, 3)}</b>.
            </p>
          </Card>
          <Callout tone={pair === "bmi_sbp" ? "warn" : undefined}>
            {pair === "age_sbp" && <>Age explains about a quarter of the variation in blood pressure. Most of what makes one person's reading differ from another's is in ε.</>}
            {pair === "bmi_sbp" && <>BMI alone explains almost nothing about one adult's blood pressure here (R² ≈ {fmt(R2, 3)}). A population trend can be real and still predict individuals poorly.</>}
            {pair === "a1c_glucose" && <>HbA1c and fasting glucose move together closely. This is the kind of near-linear relationship behind the “estimated average glucose” formula.</>}
          </Callout>
        </div>
      </div>

      <h2 style={{ marginTop: 10 }}>Every model is three choices</h2>
      <div className="grid-3">
        <Card title="1 · Model family" sub="Which shapes f is allowed to take">
          <div className="small ink2">Straight line · curves · trees · a pretrained foundation model.</div>
          <div className="small" style={{ marginTop: 8 }}><a href="#/ladder">See the model ladder →</a></div>
        </Card>
        <Card title="2 · Loss" sub="How a miss is scored when picking the best f">
          <div className="small ink2">Squared error (least squares) · absolute error · a full predictive distribution.</div>
          <div className="small" style={{ marginTop: 8 }}><a href="#/line">Try both losses →</a></div>
        </Card>
        <Card title="3 · Data and split" sub="Which patients it learns from, which it is graded on">
          <div className="small ink2">Training vs held-out patients · one row per visit or per patient. This is where most mistakes hide.</div>
          <div className="small" style={{ marginTop: 8 }}><a href="#/splits">See why the split matters →</a></div>
        </Card>
      </div>
      <SourceNote>Linear regression = straight-line family + squared-error loss. Everything else in this walk-through is a different answer to choice 1, and choice 3 decides whether you can trust the score.</SourceNote>
    </div>
  );
}

import { useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { HoverColumns, Legend, Plot, TipRow, useWidth } from "../charts/kit";
import { Card, Callout, Loading, Slider, SourceNote } from "../components/ui";
import { fmt, PlaygroundData, TreeNode, useData } from "../data";

type Patient = { age: number; bmi: number };

function route(t: TreeNode, pt: Patient): TreeNode[] {
  const path: TreeNode[] = [t];
  let n = t;
  while (!n.leaf) {
    n = pt[n.feature] <= n.threshold ? n.left : n.right;
    path.push(n);
  }
  return path;
}

const FEAT = { age: "Age", bmi: "BMI" };
const TREE_COLOR = ["var(--m-linear)", "var(--m-gbm)", "var(--m-tabicl)"];
const AVG_COLOR = "var(--ink)";

const leafOf = (t: TreeNode, pt: Patient) => (route(t, pt).at(-1) as Extract<TreeNode, { leaf: true }>).value;

/** Draws a depth-2 tree as a flowchart, highlighting the path this patient takes. */
function TreeDiagram({ tree, patient, index }: { tree: TreeNode; patient: Patient; index: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const path = new Set(route(tree, patient));
  const W = Math.max(260, width);
  const H = 220;
  const nodes: { n: TreeNode; x: number; y: number; parent?: { x: number; y: number }; side?: "yes" | "no" }[] = [];
  const walk = (n: TreeNode, x: number, y: number, span: number, parent?: { x: number; y: number }, side?: "yes" | "no") => {
    nodes.push({ n, x, y, parent, side });
    if (!n.leaf) {
      walk(n.left, x - span / 2, y + 82, span / 2, { x, y }, "yes");
      walk(n.right, x + span / 2, y + 82, span / 2, { x, y }, "no");
    }
  };
  walk(tree, W / 2, 26, W / 2);
  const leaf = route(tree, patient).at(-1) as Extract<TreeNode, { leaf: true }>;
  return (
    <div ref={ref}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="row" style={{ gap: 6 }}><span className="swatch" style={{ background: TREE_COLOR[index] }} /><b className="small">Tree {index + 1}</b></span>
        <span className="small">predicts <b className="tnum" style={{ color: "var(--m-rf)" }}>{fmt(leaf.value, 0)}</b></span>
      </div>
      <svg width={W} height={H} role="img" aria-label={`Decision tree ${index + 1}`}>
        {nodes.filter((d) => d.parent).map((d, i) => {
          const on = path.has(d.n);
          return (
            <g key={`e${i}`}>
              <line x1={d.parent!.x} y1={d.parent!.y + 16} x2={d.x} y2={d.y - 16} stroke={on ? "var(--m-rf)" : "var(--axis)"} strokeWidth={on ? 3 : 1.5} />
              <text x={(d.parent!.x + d.x) / 2 + (d.side === "yes" ? -8 : 8)} y={(d.parent!.y + d.y) / 2} fontSize={11} fill="var(--muted)"
                textAnchor={d.side === "yes" ? "end" : "start"}>{d.side}</text>
            </g>
          );
        })}
        {nodes.map((d, i) => {
          const on = path.has(d.n);
          const label = d.n.leaf ? fmt(d.n.value, 0) : `${FEAT[d.n.feature]} ≤ ${fmt(d.n.threshold, 1)}?`;
          const bw = d.n.leaf ? 56 : Math.min(W / 2 - 8, 124);
          return (
            <g key={i} className="tree-node">
              <rect x={d.x - bw / 2} y={d.y - 16} width={bw} height={32} rx={8}
                fill={on ? (d.n.leaf ? "var(--m-rf)" : "color-mix(in srgb, var(--m-rf) 18%, var(--surface))") : "var(--surface-2)"} />
              <text x={d.x} y={d.y} dy="0.33em" textAnchor="middle" fontSize={12.5} fontWeight={on ? 700 : 500}
                fill={on && d.n.leaf ? "#fff" : "var(--ink)"} className="tnum">{label}</text>
              {d.n.leaf && <text x={d.x} y={d.y + 30} textAnchor="middle" fontSize={10.5} fill="var(--muted)">n={d.n.n}</text>}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Each tree's prediction (and their average) as one predictor sweeps its range, the other held at the patient's value. */
function TreeLines({ trees, patient, feature }: { trees: TreeNode[]; patient: Patient; feature: keyof Patient }) {
  const [lo, hi, step] = feature === "age" ? [18, 80, 0.1] : [16, 54, 0.05];
  const xs = Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step);
  const preds = trees.map((t) => xs.map((v) => leafOf(t, { ...patient, [feature]: v })));
  const avg = xs.map((_, i) => preds.reduce((s, p) => s + p[i], 0) / preds.length);
  const all = preds.flat();
  const ylo = Math.min(...all), yhi = Math.max(...all), pad = Math.max(2, (yhi - ylo) * 0.1);
  const other = feature === "age" ? `BMI held at ${fmt(patient.bmi, 1)}` : `age held at ${patient.age}`;
  return (
    <div>
      <div className="small muted" style={{ marginBottom: 4 }}>Sweep {feature === "age" ? "age" : "BMI"}, {other}</div>
      <Plot height={260} xDomain={[lo, hi]} yDomain={[ylo - pad, yhi + pad]} nice={false}
        xLabel={feature === "age" ? "Age (years)" : "BMI"} yLabel="Predicted SBP (mmHg)"
        ariaLabel={`Predicted SBP from each of the three trees and their average as ${FEAT[feature]} varies`}>
        {(ctx) => {
          const { x, y, h } = ctx;
          const mk = line<number>().x((_, i) => x(xs[i])).y((v) => y(v));
          return (
            <g>
              <line x1={x(patient[feature])} x2={x(patient[feature])} y1={0} y2={h} stroke="var(--m-rf)" strokeWidth={1.5} strokeDasharray="4 3" />
              {preds.map((p, i) => (
                <path key={i} d={mk(p) ?? ""} fill="none" stroke={TREE_COLOR[i]} strokeWidth={2} opacity={0.85} strokeLinejoin="round" />
              ))}
              <path d={mk(avg) ?? ""} fill="none" stroke={AVG_COLOR} strokeWidth={3.2} strokeLinejoin="round" />
              <HoverColumns xs={xs} ctx={ctx} render={(k) => (
                <>
                  <TipRow label={<b>{FEAT[feature]} {fmt(xs[k], 1)}</b>} value="" />
                  {preds.map((p, i) => <TipRow key={i} color={TREE_COLOR[i]} label={`Tree ${i + 1}`} value={fmt(p[k], 0)} />)}
                  <TipRow color={AVG_COLOR} label="Average" value={fmt(avg[k], 1)} />
                </>
              )} />
            </g>
          );
        }}
      </Plot>
    </div>
  );
}

function ForestMap({ data, patient }: { data: PlaygroundData["trees"]; patient: Patient }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const g = data.forest_grid;
  const H = 300;
  const m = { l: 48, r: 12, t: 8, b: 40 };
  const w = width - m.l - m.r, h = H - m.t - m.b;
  const x = scaleLinear().domain([g.ages[0] - 1, g.ages[g.ages.length - 1] + 1]).range([0, w]);
  const y = scaleLinear().domain([g.bmis[0] - 1, g.bmis[g.bmis.length - 1] + 1]).range([h, 0]);
  const flat = g.pred.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const col = (v: number) => `color-mix(in srgb, var(--seq-7) ${Math.round(((v - lo) / (hi - lo)) * 100)}%, var(--seq-1))`;
  const cw = w / g.ages.length, ch = h / g.bmis.length;
  return (
    <div ref={ref}>
      <svg width={width} height={H} role="img" aria-label="Heatmap of the 300-tree forest's predictions by age and BMI">
        <g transform={`translate(${m.l},${m.t})`}>
          {g.ages.map((a, i) => g.bmis.map((b, j) => (
            <rect key={`${i}-${j}`} x={x(a) - cw / 2} y={y(b) - ch / 2} width={cw + 0.5} height={ch + 0.5} fill={col(g.pred[i][j])} />
          )))}
          {data.points.age.map((a, i) => (a < g.ages[0] || a > g.ages[g.ages.length - 1] || data.points.bmi[i] < g.bmis[0] || data.points.bmi[i] > g.bmis[g.bmis.length - 1]) ? null : (
            <circle key={i} cx={x(a)} cy={y(data.points.bmi[i])} r={1.6} fill="var(--ink)" opacity={0.35} />
          ))}
          <circle cx={x(patient.age)} cy={y(patient.bmi)} r={8} fill="none" stroke="var(--m-rf)" strokeWidth={3} />
          <circle cx={x(patient.age)} cy={y(patient.bmi)} r={2.5} fill="var(--m-rf)" />
          {[20, 30, 40, 50, 60, 70, 80].map((t) => (
            <text key={t} x={x(t)} y={h + 16} textAnchor="middle" fontSize={11} fill="var(--muted)">{t}</text>
          ))}
          {[20, 30, 40, 50].map((t) => (
            <text key={t} x={-8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)">{t}</text>
          ))}
          <text x={w / 2} y={h + 34} textAnchor="middle" className="axis-label">Age (years)</text>
          <text transform={`translate(${-36},${h / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">BMI</text>
        </g>
      </svg>
      <div className="row small muted" style={{ gap: 8 }}>
        <span>{fmt(lo, 0)} mmHg</span>
        <span style={{ width: 120, height: 10, borderRadius: 4, background: "linear-gradient(90deg, var(--seq-1), var(--seq-7))" }} />
        <span>{fmt(hi, 0)} mmHg</span>
      </div>
    </div>
  );
}

export default function Forest() {
  const pg = useData<PlaygroundData>("playground");
  const [pt, setPt] = useState<Patient>({ age: 55, bmi: 28 });
  if (!pg) return <Loading />;
  const t = pg.trees;
  const leafVals = t.trees.map((tr) => leafOf(tr, pt));
  const avg3 = leafVals.reduce((a, b) => a + b, 0) / leafVals.length;
  const g = t.forest_grid;
  const ai = g.ages.reduce((b, a, i) => (Math.abs(a - pt.age) < Math.abs(g.ages[b] - pt.age) ? i : b), 0);
  const bi = g.bmis.reduce((b, v, i) => (Math.abs(v - pt.bmi) < Math.abs(g.bmis[b] - pt.bmi) ? i : b), 0);
  const full = g.pred[ai][bi];

  return (
    <div className="stack">
      <Card>
        <div className="controls-bar" style={{ marginBottom: 0 }}>
          <Slider label="Patient age" value={pt.age} min={18} max={80} onChange={(v) => setPt({ ...pt, age: v })} format={(v) => `${v} y`} />
          <Slider label="Patient BMI" value={pt.bmi} min={16} max={54} step={0.5} onChange={(v) => setPt({ ...pt, bmi: v })} format={(v) => fmt(v, 1)} />
        </div>
      </Card>
      <div className="grid-3">
        {t.trees.map((tr, i) => (
          <Card key={i}><TreeDiagram tree={tr} patient={pt} index={i} /></Card>
        ))}
      </div>
      <Card title="Three trees, and their average" sub="Each tree's prediction is a staircase: flat within a leaf, jumping at each split. Averaging them gives more, smaller steps. The dashed line is your patient.">
        <Legend items={[
          ...t.trees.map((_, i) => ({ label: `Tree ${i + 1}`, color: TREE_COLOR[i] })),
          { label: "Average of 3 trees", color: AVG_COLOR },
          { label: "Your patient", color: "var(--m-rf)", kind: "dash" as const },
        ]} />
        <div className="grid-2">
          <TreeLines trees={t.trees} patient={pt} feature="age" />
          <TreeLines trees={t.trees} patient={pt} feature="bmi" />
        </div>
      </Card>
      <div className="split-wide">
        <Card title="The full forest: 300 trees" sub={`Predicted SBP by age and BMI, trained on ${t.n} NHANES adults (dots). The ring is your patient.`}>
          <ForestMap data={t} patient={pt} />
        </Card>
        <div className="stack">
          <div className="grid-2">
            <div className="card">
              <div className="stat-label">Average of 3 trees</div>
              <div className="stat-value tnum">{fmt(avg3, 0)}</div>
              <div className="stat-note">({leafVals.map((v) => fmt(v, 0)).join(" + ")}) ÷ 3</div>
            </div>
            <div className="card">
              <div className="stat-label">Forest of 300</div>
              <div className="stat-value tnum" style={{ color: "var(--m-rf)" }}>{fmt(full, 0)}</div>
              <div className="stat-note">mmHg, average over 300 trees</div>
            </div>
          </div>
          <Card>
            <ol style={{ margin: 0, paddingLeft: 20 }} className="small">
              <li><b>One tree</b> is a flowchart of yes/no questions. Each leaf predicts the average of the training patients that land there.</li>
              <li><b>Grow hundreds</b>, each on a random resample of patients and a random subset of predictors.</li>
              <li><b>Average their answers.</b> Each tree is wrong in its own way, so many of the mistakes cancel.</li>
            </ol>
          </Card>
          <Callout>
            Look at the map's edges: beyond the oldest or heaviest patients in the data, the colour stops changing. Forests <b>can't extrapolate</b>.
            Predictions flatten at the edge of the training data.
          </Callout>
        </div>
      </div>
      <div className="grid-2">
        <Callout tone="good"><b>Reach for it when</b> relationships curve or interact, predictors are mixed, and you want strong results with little tuning.</Callout>
        <Callout tone="warn"><b>Watch out for</b> no readable equation, no extrapolation, and memorising the training data (training R² far above held-out R²).</Callout>
      </div>
      <SourceNote>
        In the clinic: a random forest on primary-care records from 61,740 patients predicted future eGFR (R² = 0.95), but current eGFR was a predictor.
        Always ask how much the forest adds beyond the obvious predictor. Zhao J et al. Math Biosci 2019;310:24–30.
        The three trees shown are real trees (depth 2) from a forest fit to these data.
      </SourceNote>
    </div>
  );
}

import { useState } from "react";
import { TrainTestBars, TrainTestLegend } from "../charts/bars";
import { Card, Callout, Loading, ModelPicker, Segmented, SourceNote } from "../components/ui";
import { fmt, MODEL_COLOR, MODEL_IDS, MODEL_LABEL, ModelId, SplitId, useDataset } from "../data";

const SPLITS: { id: SplitId; letter: string; title: string; verdict: string; question: string }[] = [
  { id: "random", letter: "A", title: "Random rows", verdict: "Inflated", question: "Tested on recordings minutes apart from ones it trained on." },
  { id: "temporal", letter: "B", title: "Track known patients", verdict: "Honest for monitoring", question: "Train on each patient's earlier visits, test on their later ones." },
  { id: "group", letter: "C", title: "Screen new patients", verdict: "Honest for new patients", question: "Every test patient is someone the model has never seen." },
];

/** Illustrative timeline: 3 patients × 18 recordings, coloured by the role each split gives them in one fold. */
function Timeline({ split }: { split: SplitId }) {
  const rows = 3, n = 18;
  const role = (p: number, t: number) => {
    if (split === "random") return (p * 7 + t * 11) % 5 === 0 ? "test" : "train";
    if (split === "temporal") return t < 12 ? "train" : t < 15 ? "test" : "unused";
    return p === 1 ? "test" : "train";
  };
  const color = { train: "var(--m-linear)", test: "var(--m-rf)", unused: "var(--grid)" } as const;
  return (
    <div role="img" aria-label="Diagram of which recordings are used for training and testing">
      {Array.from({ length: rows }, (_, p) => (
        <div key={p} className="row" style={{ gap: 3, flexWrap: "nowrap", marginBottom: 4 }}>
          <span className="small muted" style={{ width: 70, flex: "none" }}>Patient {p + 1}</span>
          {Array.from({ length: n }, (_, t) => (
            <span key={t} style={{ flex: 1, height: 14, borderRadius: 3, background: color[role(p, t)] }} />
          ))}
        </div>
      ))}
      <div className="small muted" style={{ textAlign: "right" }}>time →</div>
    </div>
  );
}

export default function Splits() {
  const ds = useDataset("parkinsons");
  const [model, setModel] = useState<ModelId>("rf");
  const [split, setSplit] = useState<SplitId>("random");
  if (!ds) return <Loading />;

  return (
    <div className="stack">
      <div className="controls-bar">
        <ModelPicker value={model} onChange={setModel} models={["linear", "rf", "gbm", "tabicl"]} label="Big numbers show" />
      </div>
      <div className="grid-3">
        {SPLITS.map((s) => {
          const m = ds.cv[s.id]!.models[model];
          const active = s.id === split;
          return (
            <button key={s.id} className="card card-btn" onClick={() => setSplit(s.id)} aria-pressed={active}
              style={{ textAlign: "left", cursor: "pointer", borderColor: active ? "var(--accent)" : undefined, borderWidth: active ? 2 : 1 }}>
              <div className="row" style={{ gap: 8 }}><span className="badge-num">{s.letter}</span><b>{s.title}</b></div>
              <div className="stat-value huge tnum" style={{ color: m.mean_test_r2 < 0 ? "var(--critical)" : "var(--ink)", marginTop: 8 }}>{fmt(m.mean_test_r2, 2)}</div>
              <div className="small"><b>{s.verdict}.</b> <span className="ink2">{s.question}</span></div>
              <div className="small muted" style={{ marginTop: 4 }}>{MODEL_LABEL[model]} · held-out R² ± {fmt(m.sd_test_r2, 2)}</div>
            </button>
          );
        })}
      </div>

      <div className="split-wide">
        <Card title={`Split ${SPLITS.find((s) => s.id === split)!.letter}: every model`} sub={ds.cv[split]!.description}>
          <TrainTestLegend />
          <TrainTestBars rows={MODEL_IDS.map((m) => {
            const mm = ds.cv[split]!.models[m];
            return { id: m, label: MODEL_LABEL[m], color: MODEL_COLOR[m], train: mm.mean_train_r2, test: mm.mean_test_r2, testSd: mm.sd_test_r2 };
          })} />
        </Card>
        <div className="stack">
          <Card title="Who is in the test set?" sub="One fold, illustrated">
            <Segmented value={split} onChange={setSplit} options={SPLITS.map((s) => ({ value: s.id, label: s.letter }))} ariaLabel="Split" />
            <div className="spacer" />
            <Timeline split={split} />
            <div className="legend">
              <span className="legend-item"><span className="swatch" style={{ background: "var(--m-linear)" }} />Train</span>
              <span className="legend-item"><span className="swatch" style={{ background: "var(--m-rf)" }} />Test</span>
              {split === "temporal" && <span className="legend-item"><span className="swatch" style={{ background: "var(--grid)" }} />Future, not yet used</span>}
            </div>
          </Card>
          <Callout>
            {split === "random" && <>A recording's nearest neighbours in time sit in the training set, so the model can nearly copy its own patient's score. This answers no clinical question.</>}
            {split === "temporal" && <>Honest if the model will <b>monitor patients it already knows</b>: it learns each person's level from early visits. Use <code>TimeSeriesSplit</code> logic per patient.</>}
            {split === "group" && <>Honest if the model will <b>screen new patients</b>: <code>GroupKFold(groups=patient_id)</code>. Here no model beats the average, so the voice measures don't generalise across people.</>}
          </Callout>
        </div>
      </div>
      <SourceNote>
        {ds.source}: {ds.n_rows.toLocaleString()} voice recordings from {ds.n_patients} patients over about six months. Temporal split: each patient's recordings in time
        order are cut into 6 blocks; fold k trains on blocks before k and tests on block k.
      </SourceNote>
    </div>
  );
}

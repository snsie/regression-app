import { scaleLinear } from "d3-scale";
import { ReactNode } from "react";
import { fmt } from "../data";
import { TipRow, Tooltip, useTip, useWidth } from "./kit";

export interface BarRow {
  id: string;
  label: ReactNode;
  color: string;
  train?: number | null;
  test: number;
  testSd?: number;
}

/**
 * Horizontal bars: held-out score as a solid bar, training score as an outlined
 * ghost bar behind it, so the overfitting gap is the visible difference.
 */
export function TrainTestBars({
  rows, domain, showTrain = true, valueLabel = "R²", digits = 2, rowHeight = 46, labelWidth = 170,
}: {
  rows: BarRow[];
  domain?: [number, number];
  showTrain?: boolean;
  valueLabel?: string;
  digits?: number;
  rowHeight?: number;
  labelWidth?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const { tip, show, hide } = useTip();
  const lw = width < 520 ? 110 : labelWidth;
  const vals = rows.flatMap((r) => [r.test - (r.testSd ?? 0), r.test + (r.testSd ?? 0), showTrain && r.train != null ? r.train : r.test]);
  const lo = domain?.[0] ?? Math.min(0, ...vals);
  const hi = domain?.[1] ?? Math.max(1, ...vals);
  const right = 64;
  const x = scaleLinear().domain([lo, hi]).range([lw, Math.max(lw + 40, width - right)]);
  const height = rows.length * rowHeight + 30;
  const ticks = x.ticks(Math.max(3, Math.floor((width - lw) / 90)));
  const x0 = x(Math.max(lo, Math.min(hi, 0)));
  return (
    <div className="chart" ref={ref} onMouseLeave={hide}>
      <svg width={width} height={height} role="img" aria-label={`Bar chart of ${valueLabel} by model`}>
        {ticks.map((t) => (
          <g key={t} className="tick">
            <line x1={x(t)} x2={x(t)} y1={0} y2={rows.length * rowHeight} />
            <text x={x(t)} y={rows.length * rowHeight + 18} textAnchor="middle">{fmt(t, t % 1 === 0 ? 0 : 1)}</text>
          </g>
        ))}
        <line x1={x0} x2={x0} y1={0} y2={rows.length * rowHeight} stroke="var(--ink-2)" strokeWidth={1.2} />
        {rows.map((r, i) => {
          const cy = i * rowHeight + rowHeight / 2;
          const bh = Math.min(22, rowHeight * 0.5);
          const tx = x(Math.max(lo, Math.min(hi, r.test)));
          const trx = r.train != null ? x(Math.max(lo, Math.min(hi, r.train))) : null;
          const sdx = r.testSd != null ? x(Math.min(hi, r.test + r.testSd)) : tx;
          const labelX = Math.max(tx, sdx, showTrain ? trx ?? tx : tx, x0) + 8;
          return (
            <g
              key={r.id}
              onMouseMove={(e) => {
                const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                show(e.clientX - box.left, cy - bh / 2, (
                  <>
                    <TipRow color={r.color} label={<b>{r.label}</b>} value="" />
                    <TipRow label={`Held-out ${valueLabel}`} value={fmt(r.test, digits) + (r.testSd != null ? ` ± ${fmt(r.testSd, digits)}` : "")} />
                    {r.train != null && showTrain && <TipRow label={`Training ${valueLabel}`} value={fmt(r.train, digits)} />}
                  </>
                ));
              }}
            >
              <rect x={0} y={i * rowHeight} width={width} height={rowHeight} fill="transparent" />
              <text x={lw - 10} y={cy} dy="0.32em" textAnchor="end" fill="var(--ink-2)" fontSize={13}>
                {r.label}
              </text>
              {showTrain && trx != null && (
                <rect
                  x={Math.min(x0, trx)} y={cy - bh / 2 - 3} width={Math.abs(trx - x0)} height={bh + 6} rx={4}
                  fill="none" stroke={r.color} strokeWidth={1.5} strokeDasharray="4 3"
                />
              )}
              <rect x={Math.min(x0, tx)} y={cy - bh / 2} width={Math.max(1, Math.abs(tx - x0))} height={bh} rx={4} fill={r.color} />
              {r.testSd != null && (
                <line x1={x(r.test - r.testSd)} x2={x(r.test + r.testSd)} y1={cy} y2={cy} stroke="var(--ink)" strokeWidth={1.5} opacity={0.7} />
              )}
              <text x={labelX} y={cy} dy="0.32em" fontSize={13} fontWeight={650} fill="var(--ink)" className="tnum">
                {fmt(r.test, digits)}
              </text>
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

export function TrainTestLegend({ valueLabel = "R²", sd = true }: { valueLabel?: string; sd?: boolean }) {
  return (
    <div className="legend">
      <span className="legend-item">
        <span className="swatch" style={{ background: "var(--ink-2)" }} /> Held-out (cross-validated) {valueLabel}
      </span>
      <span className="legend-item">
        <span className="swatch" style={{ border: "1.5px dashed var(--ink-2)", background: "transparent" }} /> Training {valueLabel}
      </span>
      {sd && (
        <span className="legend-item">
          <span className="legend-line" style={{ background: "var(--ink)", height: 2 }} /> ± 1 SD across folds
        </span>
      )}
    </div>
  );
}

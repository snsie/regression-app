import { ReactNode, useCallback, useLayoutEffect, useRef, useState } from "react";
import { scaleLinear, scaleLog, ScaleLinear, ScaleLogarithmic } from "d3-scale";

export type Scale = ScaleLinear<number, number> | ScaleLogarithmic<number, number>;

export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(600);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

export interface TipState { x: number; y: number; content: ReactNode }

export function useTip() {
  const [tip, setTip] = useState<TipState | null>(null);
  const show = useCallback((x: number, y: number, content: ReactNode) => setTip({ x, y, content }), []);
  const hide = useCallback(() => setTip(null), []);
  return { tip, show, hide };
}

export function Tooltip({ tip }: { tip: TipState | null }) {
  if (!tip) return null;
  return (
    <div className="tooltip" style={{ left: tip.x, top: tip.y }}>
      {tip.content}
    </div>
  );
}

export function TipRow({ color, label, value }: { color?: string; label: ReactNode; value: ReactNode }) {
  return (
    <div className="tt-row">
      <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        {color && <span className="swatch" style={{ background: color }} />}
        {label}
      </span>
      <span>{value}</span>
    </div>
  );
}

export interface PlotCtx {
  x: Scale;
  y: Scale;
  w: number; // inner width
  h: number; // inner height
  show: (x: number, y: number, content: ReactNode) => void; // x,y in inner coords
  hide: () => void;
}

export interface PlotProps {
  height?: number;
  margin?: { top?: number; right?: number; bottom?: number; left?: number };
  xDomain: [number, number];
  yDomain: [number, number];
  xLog?: boolean;
  xLabel?: string;
  yLabel?: string;
  xTicks?: number | number[];
  yTicks?: number | number[];
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  ariaLabel: string;
  children: (ctx: PlotCtx) => ReactNode;
  nice?: boolean;
  zeroLine?: boolean;
}

const fmtTick = (v: number) => {
  const a = Math.abs(v);
  const s = a >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 0 }) : a >= 10 || v === 0 ? String(Math.round(v * 10) / 10) : String(Math.round(v * 100) / 100);
  return s.replace(/^-/, "−");
};

/** A responsive SVG plot with gridlines, axes and a hover-tooltip layer. */
export function Plot(p: PlotProps) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const { tip, show, hide } = useTip();
  const height = p.height ?? 320;
  const m = { top: 12, right: 16, bottom: p.xLabel ? 46 : 28, left: p.yLabel ? 62 : 46, ...p.margin };
  const w = Math.max(10, width - m.left - m.right);
  const h = Math.max(10, height - m.top - m.bottom);
  const x = (p.xLog ? scaleLog() : scaleLinear()).domain(p.xDomain).range([0, w]);
  const y = scaleLinear().domain(p.yDomain).range([h, 0]);
  if (p.nice !== false && !p.xLog) (x as ScaleLinear<number, number>).nice();
  if (p.nice !== false) y.nice();
  const xt = Array.isArray(p.xTicks) ? p.xTicks : x.ticks(typeof p.xTicks === "number" ? p.xTicks : Math.max(3, Math.floor(w / 90)));
  const yt = Array.isArray(p.yTicks) ? p.yTicks : y.ticks(typeof p.yTicks === "number" ? p.yTicks : Math.max(3, Math.floor(h / 55)));
  const xf = p.xFormat ?? fmtTick;
  const yf = p.yFormat ?? fmtTick;
  const ctx: PlotCtx = {
    x, y, w, h,
    show: (cx, cy, c) => show(cx + m.left, cy + m.top, c),
    hide,
  };
  return (
    <div className="chart" ref={ref} onMouseLeave={hide}>
      <svg width={width} height={height} role="img" aria-label={p.ariaLabel}>
        <g transform={`translate(${m.left},${m.top})`}>
          {yt.map((t) => (
            <g key={`y${t}`} className="tick" transform={`translate(0,${y(t)})`}>
              <line x2={w} />
              <text x={-8} dy="0.32em" textAnchor="end">{yf(t)}</text>
            </g>
          ))}
          {xt.map((t) => (
            <g key={`x${t}`} className="tick" transform={`translate(${x(t)},${h})`}>
              <line y1={0} y2={5} />
              <text y={18} textAnchor="middle">{xf(t)}</text>
            </g>
          ))}
          <line className="axis-line" x1={0} x2={w} y1={h} y2={h} />
          {p.zeroLine && y.domain()[0] < 0 && y.domain()[1] > 0 && (
            <line x1={0} x2={w} y1={y(0)} y2={y(0)} stroke="var(--ink-2)" strokeWidth={1} strokeDasharray="3 3" />
          )}
          {p.children(ctx)}
          {p.xLabel && (
            <text className="axis-label" x={w / 2} y={h + 40} textAnchor="middle">{p.xLabel}</text>
          )}
          {p.yLabel && (
            <text className="axis-label" transform={`translate(${-m.left + 16},${h / 2}) rotate(-90)`} textAnchor="middle">{p.yLabel}</text>
          )}
        </g>
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; kind?: "line" | "dot" | "box" | "dash" }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span className="legend-item" key={it.label}>
          {it.kind === "dot" ? (
            <span className="dot" style={{ background: it.color }} />
          ) : it.kind === "box" ? (
            <span className="swatch" style={{ background: it.color }} />
          ) : it.kind === "dash" ? (
            <span className="legend-line" style={{ background: `repeating-linear-gradient(90deg, ${it.color} 0 5px, transparent 5px 8px)` }} />
          ) : (
            <span className="legend-line" style={{ background: it.color }} />
          )}
          {it.label}
        </span>
      ))}
    </div>
  );
}

/** Invisible vertical hover strips that snap to the nearest x value — a crosshair for line charts. */
export function HoverColumns({
  xs, ctx, render,
}: { xs: number[]; ctx: PlotCtx; render: (i: number) => ReactNode }) {
  const [active, setActive] = useState<number | null>(null);
  const px = xs.map((v) => ctx.x(v));
  return (
    <g>
      {active != null && <line x1={px[active]} x2={px[active]} y1={0} y2={ctx.h} stroke="var(--axis)" strokeWidth={1} />}
      <rect
        width={ctx.w}
        height={ctx.h}
        fill="transparent"
        onMouseMove={(e) => {
          const box = (e.currentTarget as SVGRectElement).getBoundingClientRect();
          const mx = e.clientX - box.left;
          let best = 0;
          for (let i = 1; i < px.length; i++) if (Math.abs(px[i] - mx) < Math.abs(px[best] - mx)) best = i;
          setActive(best);
          ctx.show(px[best], Math.max(30, e.clientY - box.top), render(best));
        }}
        onMouseLeave={() => {
          setActive(null);
          ctx.hide();
        }}
      />
    </g>
  );
}

export const extent = (vals: number[], pad = 0.05): [number, number] => {
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) if (v != null && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = hi - lo || 1;
  return [lo - span * pad, hi + span * pad];
};

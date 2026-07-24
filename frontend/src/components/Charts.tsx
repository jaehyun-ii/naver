import { useState } from "react";
import { useTheme } from "@mui/material/styles";
import { Box, Typography } from "@mui/material";

// 의존성 없이 SVG로 그리는 경량 차트들. 콘솔의 소규모 시계열/막대 표시에 충분하다.
// 색은 job으로: 크기=순차(단일 hue), 상태=예약색(성공/경고/위험). 축·그리드는 recessive.

export function LineChart({ values, color, ylabel }: { values: number[]; color?: string; ylabel?: string }) {
  const theme = useTheme();
  const stroke = color ?? theme.palette.error.main;
  const W = 480, H = 120, P = 8;
  if (!values.length) return null;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => {
    const x = P + (i / Math.max(values.length - 1, 1)) * (W - 2 * P);
    const y = H - P - ((v - min) / span) * (H - 2 * P);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <Box sx={{ mt: 1 }}>
      {ylabel && <Typography variant="caption" color="text.secondary">{ylabel}</Typography>}
      <Box component="svg" viewBox={`0 0 ${W} ${H}`} sx={{ width: "100%", maxWidth: W, display: "block", bgcolor: "background.default", borderRadius: 1, border: 1, borderColor: "divider" }}>
        <polyline points={pts} fill="none" stroke={stroke} strokeWidth={2} />
      </Box>
      <Typography variant="caption" color="text.secondary">min {min.toFixed(3)} · max {max.toFixed(3)}</Typography>
    </Box>
  );
}

export interface BarDatum { label: string; value: number; threshold?: number }

export function BarChart({ data, fmt }: { data: BarDatum[]; fmt?: (v: number) => string }) {
  const theme = useTheme();
  const format = fmt ?? ((v: number) => String(v));
  if (!data.length) return null;
  const max = Math.max(...data.map((d) => Math.max(d.value, d.threshold ?? 0)), 0.0001);
  return (
    <Box sx={{ mt: 1, display: "flex", flexDirection: "column", gap: 1 }}>
      {data.map((d, i) => {
        const over = d.threshold != null && d.value > d.threshold;
        const barColor = d.threshold != null
          ? (over ? theme.palette.error.main : theme.palette.success.main)
          : theme.palette.primary.main;
        return (
          <Box key={i} sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Typography variant="caption" sx={{ width: 90, flexShrink: 0 }} noWrap>{d.label}</Typography>
            <Box sx={{ position: "relative", flex: 1, height: 16, bgcolor: "background.default", borderRadius: 1, border: 1, borderColor: "divider" }}>
              <Box sx={{ width: `${(d.value / max) * 100}%`, height: "100%", bgcolor: barColor, borderRadius: 1 }} />
              {d.threshold != null && (
                <Box sx={{ position: "absolute", top: -2, bottom: -2, left: `${(d.threshold / max) * 100}%`, borderLeft: "2px dashed", borderColor: "text.secondary" }} />
              )}
            </Box>
            <Typography variant="caption" sx={{ width: 56, flexShrink: 0, textAlign: "right" }}>{format(d.value)}</Typography>
          </Box>
        );
      })}
    </Box>
  );
}

// ── 산점도 — 탐색/실험 포인트(예: HPO trial score). 크기=순차 강조, best=상태색. ──
export interface ScatterPoint { x: number; y: number; best?: boolean; tooltip?: string[] }
export function ScatterPlot({ points, xlabel, ylabel, fmtX, fmtY }: {
  points: ScatterPoint[]; xlabel?: string; ylabel?: string;
  fmtX?: (v: number) => string; fmtY?: (v: number) => string;
}) {
  const theme = useTheme();
  const [hover, setHover] = useState<number | null>(null);
  const W = 560, H = 260, ML = 46, MB = 34, MT = 14, MR = 14;
  const fX = fmtX ?? ((v: number) => String(v));
  const fY = fmtY ?? ((v: number) => v.toFixed(2));
  if (!points.length) return null;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const xspan = (xmax - xmin) || 1, yspan = (ymax - ymin) || 1;
  const px = (x: number) => ML + ((x - xmin) / xspan) * (W - ML - MR);
  const py = (y: number) => H - MB - ((y - ymin) / yspan) * (H - MB - MT);
  const grid = theme.palette.divider, ink = theme.palette.text.secondary, dot = theme.palette.primary.main;
  const surface = theme.palette.background.paper, best = theme.palette.success.main;
  const yticks = Array.from({ length: 4 }, (_, i) => ymin + (yspan * i) / 3);
  return (
    <Box sx={{ mt: 1 }}>
      <Box component="svg" viewBox={`0 0 ${W} ${H}`} sx={{ width: "100%", display: "block" }} role="img" aria-label="산점도">
        {yticks.map((t, i) => (
          <g key={i}>
            <line x1={ML} x2={W - MR} y1={py(t)} y2={py(t)} stroke={grid} strokeWidth={1} />
            <text x={ML - 6} y={py(t) + 3} textAnchor="end" fontSize={10} fill={ink}>{fY(t)}</text>
          </g>
        ))}
        <line x1={ML} x2={ML} y1={MT} y2={H - MB} stroke={grid} strokeWidth={1} />
        <text x={ML} y={H - 6} fontSize={10} fill={ink}>{fX(xmin)}</text>
        <text x={W - MR} y={H - 6} textAnchor="end" fontSize={10} fill={ink}>{fX(xmax)}</text>
        {xlabel && <text x={(W + ML) / 2} y={H - 6} textAnchor="middle" fontSize={10} fill={ink}>{xlabel}</text>}
        {ylabel && <text x={12} y={MT + 4} fontSize={10} fill={ink}>{ylabel}</text>}
        {points.map((p, i) => (
          <circle key={i} cx={px(p.x)} cy={py(p.y)} r={hover === i ? 7 : p.best ? 6 : 4.5}
            fill={p.best ? best : dot} fillOpacity={p.best ? 1 : 0.72}
            stroke={surface} strokeWidth={hover === i || p.best ? 2 : 0}
            style={{ cursor: "pointer" }}
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
        ))}
        {hover != null && points[hover] && (() => {
          const p = points[hover]; const lines = p.tooltip ?? [`x ${fX(p.x)}`, `y ${fY(p.y)}`];
          const tw = 132, th = 14 + lines.length * 13; let tx = px(p.x) + 10; const ty = Math.max(MT, py(p.y) - th - 8);
          if (tx + tw > W) tx = px(p.x) - tw - 10;
          return (
            <g pointerEvents="none">
              <rect x={tx} y={ty} width={tw} height={th} rx={6} fill={surface} stroke={grid} strokeWidth={1} opacity={0.98} />
              {lines.map((l, k) => (
                <text key={k} x={tx + 8} y={ty + 15 + k * 13} fontSize={11}
                  fill={k === 0 ? theme.palette.text.primary : ink} fontWeight={k === 0 ? 700 : 400}>{l}</text>
              ))}
            </g>
          );
        })()}
      </Box>
    </Box>
  );
}

// ── 히스토그램 — 단일 분포(예: 지연 ms). 단일 hue, 막대 hover로 구간·빈도 표시. ──
export function Histogram({ values, bins = 12, fmt, unit }: {
  values: number[]; bins?: number; fmt?: (v: number) => string; unit?: string;
}) {
  const theme = useTheme();
  const [hover, setHover] = useState<number | null>(null);
  const f = fmt ?? ((v: number) => Math.round(v).toString());
  if (values.length < 2) return null;
  const min = Math.min(...values), max = Math.max(...values);
  const span = (max - min) || 1, step = span / bins;
  const counts = new Array(bins).fill(0);
  values.forEach((v) => { const b = Math.min(bins - 1, Math.floor((v - min) / step)); counts[b] += 1; });
  const cmax = Math.max(...counts, 1);
  const W = 560, H = 220, ML = 34, MB = 30, MT = 12, MR = 12;
  const bw = (W - ML - MR) / bins;
  const grid = theme.palette.divider, ink = theme.palette.text.secondary, bar = theme.palette.primary.main, surface = theme.palette.background.paper;
  const median = [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
  return (
    <Box sx={{ mt: 1 }}>
      <Box component="svg" viewBox={`0 0 ${W} ${H}`} sx={{ width: "100%", display: "block" }} role="img" aria-label="분포">
        {[0.5, 1].map((r, i) => (
          <line key={i} x1={ML} x2={W - MR} y1={MT + (H - MB - MT) * (1 - r)} y2={MT + (H - MB - MT) * (1 - r)} stroke={grid} strokeWidth={1} />
        ))}
        {counts.map((c, i) => {
          const h = (c / cmax) * (H - MB - MT);
          const x = ML + i * bw, y = H - MB - h;
          return (
            <rect key={i} x={x + 1} y={y} width={Math.max(bw - 2, 1)} height={h} rx={3}
              fill={bar} fillOpacity={hover === i ? 1 : 0.8} style={{ cursor: "pointer" }}
              onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
          );
        })}
        <line x1={ML} x2={W - MR} y1={H - MB} y2={H - MB} stroke={grid} strokeWidth={1} />
        <text x={ML} y={H - 8} fontSize={10} fill={ink}>{f(min)}{unit}</text>
        <text x={W - MR} y={H - 8} textAnchor="end" fontSize={10} fill={ink}>{f(max)}{unit}</text>
        <text x={(W + ML) / 2} y={H - 8} textAnchor="middle" fontSize={10} fill={ink}>중앙값 {f(median)}{unit}</text>
        {hover != null && (() => {
          const lo = min + hover * step, hi = lo + step;
          const label = `${f(lo)}–${f(hi)}${unit ?? ""}: ${counts[hover]}건`;
          const tw = label.length * 7 + 16; let tx = ML + hover * bw + bw / 2 - tw / 2;
          tx = Math.max(ML, Math.min(tx, W - MR - tw));
          return (
            <g pointerEvents="none">
              <rect x={tx} y={MT} width={tw} height={20} rx={6} fill={surface} stroke={grid} strokeWidth={1} opacity={0.98} />
              <text x={tx + 8} y={MT + 14} fontSize={11} fill={theme.palette.text.primary}>{label}</text>
            </g>
          );
        })()}
      </Box>
    </Box>
  );
}

export function Gauge({ value, label }: { value: number; label: string }) {
  const theme = useTheme();
  const v = Math.max(0, Math.min(100, value || 0));
  const R = 34, C = 2 * Math.PI * R;
  const color = v > 90 ? theme.palette.error.main : v > 70 ? theme.palette.warning.main : theme.palette.primary.main;
  return (
    <Box sx={{ textAlign: "center", width: 96 }}>
      <Box component="svg" viewBox="0 0 80 80" sx={{ width: 80, height: 80 }}>
        <circle cx={40} cy={40} r={R} fill="none" stroke={theme.palette.divider} strokeWidth={8} />
        <circle
          cx={40} cy={40} r={R} fill="none" stroke={color} strokeWidth={8} strokeLinecap="round"
          strokeDasharray={C} strokeDashoffset={C * (1 - v / 100)} transform="rotate(-90 40 40)"
        />
        <text x={40} y={45} textAnchor="middle" fontSize={16} fill={theme.palette.text.primary}>{Math.round(v)}%</text>
      </Box>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
    </Box>
  );
}

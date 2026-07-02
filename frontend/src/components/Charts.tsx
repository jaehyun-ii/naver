import { useTheme } from "@mui/material/styles";
import { Box, Typography } from "@mui/material";

// 의존성 없이 SVG로 그리는 경량 차트들. 콘솔의 소규모 시계열/막대 표시에 충분하다.

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

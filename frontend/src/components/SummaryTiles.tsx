import type { ReactNode } from "react";
import { Link as RouterLink } from "react-router-dom";
import { Box, Button, Card, CardActionArea, CardContent, Stack, Typography } from "@mui/material";
import type { SvgIconComponent } from "@mui/icons-material";

export interface Stat {
  label: string;
  value: ReactNode;
  hint?: string;
  icon?: SvgIconComponent;
  to?: string;
  /** MUI 색상 토큰 (예: "primary.main", "success.main") */
  accent?: string;
}

function Tile({ icon: Icon, label, value, hint, to, accent = "primary.main" }: Stat) {
  const inner = (
    <CardContent sx={{ py: 2 }}>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.75 }}>
        {Icon && (
          <Box sx={{ display: "grid", placeItems: "center", width: 28, height: 28, borderRadius: 1.5, bgcolor: "action.hover", color: accent }}>
            <Icon fontSize="small" />
          </Box>
        )}
        <Typography variant="caption" sx={{ fontWeight: 700, letterSpacing: 0.4, textTransform: "uppercase", color: "text.secondary" }}>
          {label}
        </Typography>
      </Stack>
      <Typography sx={{ fontSize: 28, fontWeight: 800, lineHeight: 1.1, fontVariantNumeric: "tabular-nums" }}>{value}</Typography>
      {hint && <Typography variant="caption" color="text.secondary">{hint}</Typography>}
    </CardContent>
  );
  return (
    <Card variant="outlined" sx={{ borderRadius: 3 }}>
      {to ? <CardActionArea component={RouterLink} to={to}>{inner}</CardActionArea> : inner}
    </Card>
  );
}

/** 페이지 상단 요약 KPI 타일 행 — 도메인 핵심 수치 2~5개. */
export function SummaryTiles({ stats, sx }: { stats: Stat[]; sx?: object }) {
  const n = stats.length;
  return (
    <Box sx={{ display: "grid", gap: 2, mb: 3,
      gridTemplateColumns: { xs: "1fr 1fr", sm: `repeat(${Math.min(n, 3)},1fr)`, lg: `repeat(${n},1fr)` }, ...sx }}>
      {stats.map((s, i) => <Tile key={i} {...s} />)}
    </Box>
  );
}

/** 안내 + CTA가 있는 리치 빈상태. */
export function RichEmpty({ icon: Icon, title, hint, actionLabel, to }: {
  icon?: SvgIconComponent; title: string; hint?: string; actionLabel?: string; to?: string;
}) {
  return (
    <Box sx={{ textAlign: "center", py: 6, px: 2 }}>
      {Icon && <Box sx={{ display: "grid", placeItems: "center", width: 48, height: 48, borderRadius: 2, bgcolor: "action.hover", color: "text.disabled", mx: "auto", mb: 1.5 }}><Icon /></Box>}
      <Typography variant="body1" fontWeight={700}>{title}</Typography>
      {hint && <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, maxWidth: 420, mx: "auto" }}>{hint}</Typography>}
      {actionLabel && to && (
        <Button variant="contained" component={RouterLink} to={to} sx={{ mt: 2 }}>{actionLabel}</Button>
      )}
    </Box>
  );
}

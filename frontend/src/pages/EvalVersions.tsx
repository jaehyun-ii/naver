import {
  Box, Card, CardContent, Chip, Stack, Table, TableBody, TableCell,
  TableHead, TableRow, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { BarChart, type BarDatum } from "../components/Charts";
import { useApi } from "../hooks/useApi";

interface EvalRun {
  run_id: string;
  name: string;
  status: string;
  method: string;
  metrics: Record<string, number>;
  created_at: number | null;
}

interface EvalVersionGroup {
  fingerprint: string;
  runs: EvalRun[];
  best_score: number | null;
  num_runs: number;
}

function statusColor(status: string): "success" | "error" | "warning" | "default" {
  const s = status.toLowerCase();
  if (s.includes("success") || s.includes("done") || s.includes("complete") || s.includes("approved")) return "success";
  if (s.includes("fail") || s.includes("error") || s.includes("reject")) return "error";
  if (s.includes("run") || s.includes("pending") || s.includes("queue")) return "warning";
  return "default";
}

function fmtMetrics(metrics: Record<string, number>): string {
  const entries = Object.entries(metrics);
  if (!entries.length) return "—";
  return entries.map(([k, v]) => `${k}=${v.toFixed(3)}`).join(", ");
}

function fmtDate(ts: number | null): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString("ko-KR");
}

export default function EvalVersions() {
  const { data, loading, error, reload } = useApi<EvalVersionGroup[]>("/api/evaluation/by-version");

  const groups = data ?? [];
  const scored = groups.filter((g): g is EvalVersionGroup & { best_score: number } => g.best_score != null);
  const barData: BarDatum[] = scored.map((g) => ({
    label: g.fingerprint.slice(0, 8),
    value: g.best_score,
  }));

  return (
    <>
      <PageHeader
        title="버전별 성능"
        subtitle="데이터 버전(fingerprint)별로 학습·평가 결과를 모아 비교. 같은/다른 데이터 버전의 모델 성능 추적."
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      {error && <ErrorView message={error} />}

      {loading && !data ? (
        <Loading />
      ) : groups.length === 0 ? (
        <EmptyView message="아직 데이터 버전 기록이 없습니다. 파이프라인을 실행하세요." />
      ) : (
        <>
          {barData.length > 1 && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h3" gutterBottom>버전별 최고 점수 비교</Typography>
                <BarChart data={barData} fmt={(v) => v.toFixed(2)} />
              </CardContent>
            </Card>
          )}

          <Stack spacing={3}>
            {groups.map((g) => (
              <Card key={g.fingerprint}>
                <CardContent>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2, flexWrap: "wrap" }}>
                    <Typography
                      variant="h3"
                      sx={{ fontFamily: "ui-monospace, monospace", fontSize: 14 }}
                    >
                      {g.fingerprint.slice(0, 16)}…
                    </Typography>
                    <Chip size="small" variant="outlined" label={`runs ${g.num_runs}`} />
                    {g.best_score != null && (
                      <Chip size="small" color="success" variant="outlined" label={`best ${g.best_score.toFixed(3)}`} />
                    )}
                  </Box>

                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>run</TableCell>
                        <TableCell>이름</TableCell>
                        <TableCell>모델</TableCell>
                        <TableCell>상태</TableCell>
                        <TableCell>메트릭</TableCell>
                        <TableCell align="right">생성 시각</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {g.runs.map((r) => (
                        <TableRow key={r.run_id}>
                          <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                            {r.run_id}
                          </TableCell>
                          <TableCell>{r.name || "—"}</TableCell>
                          <TableCell>{r.method || "—"}</TableCell>
                          <TableCell>
                            <Chip size="small" label={r.status} color={statusColor(r.status)} variant="outlined" />
                          </TableCell>
                          <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                            {fmtMetrics(r.metrics)}
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="caption" color="text.secondary">{fmtDate(r.created_at)}</Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            ))}
          </Stack>
        </>
      )}
    </>
  );
}

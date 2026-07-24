import {
  Box, Card, CardContent, Chip, LinearProgress, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import MemoryOutlined from "@mui/icons-material/Memory";
import CloudQueueOutlined from "@mui/icons-material/CloudQueueOutlined";
import ModelTrainingOutlined from "@mui/icons-material/ModelTraining";
import PublicOutlined from "@mui/icons-material/PublicOutlined";
import { PageHeader } from "../components/PageHeader";
import { SummaryTiles } from "../components/SummaryTiles";
import { Gauge } from "../components/Charts";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { useApi } from "../hooks/useApi";

interface Gpu { name: string; mem_used_mb: number; mem_total_mb: number; util_pct: number; temp_c: number }
interface Svc { name: string; status: string }
interface Infra {
  gpus: Gpu[];
  services: Svc[];
  storage?: { provider?: string; endpoint?: string };
  env?: { env?: string; domain?: string };
}
interface Models { models: string[] }

function serviceColor(status: string): "success" | "error" | "warning" | "default" {
  const s = status.toLowerCase();
  if (s.includes("up") || s.includes("running") || s.includes("healthy")) return "success";
  if (s.includes("exit") || s.includes("dead") || s.includes("down")) return "error";
  if (s.includes("restart") || s.includes("starting")) return "warning";
  return "default";
}

export default function Dashboard() {
  const infra = useApi<Infra>("/api/infra");
  const models = useApi<Models>("/api/models");

  const gpus = infra.data?.gpus ?? [];
  const services = infra.data?.services ?? [];

  return (
    <>
      <PageHeader
        title="대시보드 · GPU"
        subtitle="인프라·GPU·서비스 상태"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => { infra.reload(); models.reload(); }} aria-label="새로고침">
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        }
      />

      {infra.error && <ErrorView message={infra.error} />}

      <SummaryTiles stats={[
        { label: "GPU", value: gpus.length, hint: "검출된 장치", icon: MemoryOutlined, accent: "warning.main" },
        { label: "서비스", value: services.length, hint: "Docker 서비스", icon: CloudQueueOutlined },
        { label: "등록 모델", value: models.data?.models?.length ?? "—", hint: "게이트웨이", icon: ModelTrainingOutlined, accent: "success.main" },
        { label: "환경", value: infra.data?.env?.env ?? "—", hint: infra.data?.env?.domain, icon: PublicOutlined },
      ]} />

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>GPU</Typography>
          {infra.loading && !infra.data ? (
            <Loading />
          ) : gpus.length === 0 ? (
            <EmptyView message="검출된 GPU가 없습니다 (nvidia-smi 사용 불가일 수 있음)." />
          ) : (
            <Stack spacing={2.5}>
              {gpus.map((g, i) => {
                const memPct = g.mem_total_mb ? Math.round((g.mem_used_mb / g.mem_total_mb) * 100) : 0;
                return (
                  <Box key={i} sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                    <Gauge value={g.util_pct} label="util" />
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                        <Typography variant="body2" fontWeight={600} noWrap>{g.name}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {g.temp_c}°C · {g.mem_used_mb}/{g.mem_total_mb} MB
                        </Typography>
                      </Box>
                      <Typography variant="caption" color="text.secondary">메모리 {memPct}%</Typography>
                      <LinearProgress
                        variant="determinate" value={Math.min(memPct, 100)}
                        color={memPct > 90 ? "error" : memPct > 70 ? "warning" : "primary"}
                        sx={{ height: 6, borderRadius: 2, mt: 0.25 }}
                      />
                    </Box>
                  </Box>
                );
              })}
            </Stack>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>서비스</Typography>
          {infra.loading && !infra.data ? (
            <Loading />
          ) : services.length === 0 ? (
            <EmptyView message="표시할 서비스가 없습니다." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>이름</TableCell>
                  <TableCell align="right">상태</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {services.map((s, i) => (
                  <TableRow key={i}>
                    <TableCell>{s.name}</TableCell>
                    <TableCell align="right">
                      <Chip size="small" label={s.status} color={serviceColor(s.status)} variant="outlined" />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </>
  );
}

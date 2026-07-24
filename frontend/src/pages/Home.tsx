import type { ReactNode } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  Box, Button, Card, CardActionArea, CardContent, Chip, Divider, LinearProgress,
  Stack, Typography,
} from "@mui/material";
import AccountTreeOutlined from "@mui/icons-material/AccountTreeOutlined";
import DatasetOutlined from "@mui/icons-material/DatasetOutlined";
import ChatOutlined from "@mui/icons-material/ChatOutlined";
import TuneOutlined from "@mui/icons-material/TuneOutlined";
import MemoryOutlined from "@mui/icons-material/Memory";
import ModelTrainingOutlined from "@mui/icons-material/ModelTraining";
import AssessmentOutlined from "@mui/icons-material/AssessmentOutlined";
import CloudQueueOutlined from "@mui/icons-material/CloudQueueOutlined";
import TimelineOutlined from "@mui/icons-material/TimelineOutlined";
import GavelOutlined from "@mui/icons-material/GavelOutlined";
import VpnKeyOutlined from "@mui/icons-material/VpnKeyOutlined";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import type { SvgIconComponent } from "@mui/icons-material";
import { PageHeader } from "../components/PageHeader";
import { useApi } from "../hooks/useApi";

interface Stage { name: string; title: string; status: string; detail?: string }
interface PipelineRun {
  id: string; name: string; status: string; created_at?: number;
  stages: Stage[]; artifacts: Record<string, string>;
}
interface Models { models?: { name?: string }[] }
interface Infra { gpus?: { name?: string; util?: number; mem_used?: number; mem_total?: number }[]; services?: { status?: string }[] }
interface Dataset { fingerprint: string }
interface Hpo { id: string; status: string }
interface Release { id: string; status: string; model_ref?: string; served_name?: string }
interface KeyRow { id?: string; name?: string; tenant?: string; budget_usd?: number; spent_usd?: number }
interface Baseline { id?: string; name?: string }

const RUN_COLOR: Record<string, "success" | "error" | "info" | "warning" | "default"> = {
  succeeded: "success", failed: "error", running: "info", "post-running": "info", waiting: "warning",
};
const RUN_LABEL: Record<string, string> = {
  succeeded: "성공", failed: "실패", running: "학습 중", "post-running": "배포 중", waiting: "승인 대기",
};

// ── 라이프사이클 워크플로우 맵 ──
function WorkflowMap({ steps }: { steps: { label: string; value: ReactNode; hint: string; icon: SvgIconComponent; to: string; alert?: boolean }[] }) {
  return (
    <Card variant="outlined" sx={{ borderRadius: 3, mb: 3 }}>
      <CardContent sx={{ py: 2 }}>
        <Typography variant="caption" sx={{ fontWeight: 700, letterSpacing: 0.4, textTransform: "uppercase", color: "text.secondary" }}>
          워크플로우 · 준비부터 운영까지
        </Typography>
        <Box sx={{ display: "flex", alignItems: "stretch", gap: 0, mt: 1, overflowX: "auto" }}>
          {steps.map((s, i) => {
            const Icon = s.icon;
            return (
              <Box key={s.label} sx={{ display: "flex", alignItems: "center", flex: 1, minWidth: 150 }}>
                <CardActionArea component={RouterLink} to={s.to}
                  sx={{ borderRadius: 2, p: 1.25, flex: 1, border: "1px solid", borderColor: s.alert ? "warning.main" : "divider" }}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Icon fontSize="small" sx={{ color: s.alert ? "warning.main" : "primary.main" }} />
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="caption" color="text.secondary" noWrap>{s.label}</Typography>
                      <Typography sx={{ fontSize: 20, fontWeight: 800, lineHeight: 1.1, fontVariantNumeric: "tabular-nums" }}>
                        {s.value}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" noWrap sx={{ display: "block" }}>{s.hint}</Typography>
                    </Box>
                  </Stack>
                </CardActionArea>
                {i < steps.length - 1 && <ChevronRightIcon sx={{ color: "text.disabled", mx: 0.25, flexShrink: 0 }} />}
              </Box>
            );
          })}
        </Box>
      </CardContent>
    </Card>
  );
}

// ── 도메인 헤드라인 카드 ──
function DomainCard({ icon: Icon, title, to, headline, accent = "primary.main", children }: {
  icon: SvgIconComponent; title: string; to: string; headline?: ReactNode; accent?: string; children?: ReactNode;
}) {
  return (
    <Card variant="outlined" sx={{ borderRadius: 3, height: "100%" }}>
      <CardContent>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Box sx={{ display: "grid", placeItems: "center", width: 28, height: 28, borderRadius: 1.5, bgcolor: "action.hover", color: accent }}>
              <Icon fontSize="small" />
            </Box>
            <Typography variant="body2" fontWeight={700}>{title}</Typography>
          </Stack>
          <Button size="small" component={RouterLink} to={to}>열기</Button>
        </Stack>
        {headline != null && (
          <Typography sx={{ fontSize: 26, fontWeight: 800, lineHeight: 1.1, mb: 0.5, fontVariantNumeric: "tabular-nums" }}>{headline}</Typography>
        )}
        {children}
      </CardContent>
    </Card>
  );
}

function progressOf(r: PipelineRun): number {
  const done = r.stages.filter((s) => s.status === "succeeded").length;
  return r.stages.length ? Math.round((done / r.stages.length) * 100) : 0;
}
function activeStage(r: PipelineRun): Stage | undefined {
  return r.stages.find((s) => s.status === "running") ?? [...r.stages].reverse().find((s) => s.status !== "pending");
}
function metricSummary(r: PipelineRun): string | null {
  try {
    const m = JSON.parse(r.artifacts.adapter_metrics || r.artifacts.metrics_json || "{}");
    const f1 = m.token_f1 ?? m.reference_f1;
    return typeof f1 === "number" ? `token_f1 ${f1.toFixed(3)}` : null;
  } catch { return null; }
}

const ACTIONS: { path: string; label: string; desc: string; icon: SvgIconComponent }[] = [
  { path: "/pipe", label: "새 학습 실행", desc: "SFT·DPO·GRPO 파이프라인", icon: AccountTreeOutlined },
  { path: "/data", label: "데이터셋", desc: "JSONL 검증·빌드", icon: DatasetOutlined },
  { path: "/hpo", label: "파라미터 최적화", desc: "Optuna HPO", icon: TuneOutlined },
  { path: "/play", label: "플레이그라운드", desc: "배포 모델과 대화", icon: ChatOutlined },
];

export default function Home() {
  const runsApi = useApi<PipelineRun[]>("/api/pipeline/runs");
  const models = useApi<Models>("/api/models");
  const datasets = useApi<Dataset[]>("/api/data/datasets");
  const infra = useApi<Infra>("/api/infra");
  const hpo = useApi<Hpo[]>("/api/tuning/hpo");
  const releases = useApi<Release[]>("/api/releases");
  const keys = useApi<KeyRow[]>("/api/keys");
  const baselines = useApi<Baseline[]>("/api/drift/baselines");

  const runs = runsApi.data ?? [];
  const running = runs.filter((r) => ["running", "post-running", "waiting"].includes(r.status)).length;
  const recent = [...runs].sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0)).slice(0, 6);
  const pending = (releases.data ?? []).filter((r) => r.status === "Pending");
  const failedRuns = runs.filter((r) => r.status === "failed").slice(0, 4);
  const modelList = models.data?.models ?? [];
  const gpus = infra.data?.gpus ?? [];
  const num = (v: number | undefined | null, api: { data: unknown }) => (api.data ? (v ?? 0) : "—");

  return (
    <>
      <PageHeader
        title="LLM 운영 콘솔"
        subtitle="데이터 · 학습 · 배포 · 평가를 하나의 흐름으로 — 준비부터 운영까지 한 곳에서."
        action={
          <Button variant="contained" size="large" startIcon={<ModelTrainingOutlined />} component={RouterLink} to="/pipe">
            학습 시작
          </Button>
        }
      />

      {/* 라이프사이클 워크플로우 맵 */}
      <WorkflowMap steps={[
        { label: "데이터", value: num(datasets.data?.length, datasets), hint: "데이터셋", icon: DatasetOutlined, to: "/data" },
        { label: "학습", value: num(runs.length, runsApi), hint: `진행 중 ${running}`, icon: AccountTreeOutlined, to: "/pipe" },
        { label: "평가·승인", value: num(pending.length, releases), hint: "승인 대기", icon: AssessmentOutlined, to: "/rel", alert: pending.length > 0 },
        { label: "배포", value: num(modelList.length, models), hint: "서빙 모델", icon: CloudQueueOutlined, to: "/serving" },
        { label: "모니터링", value: num(baselines.data?.length, baselines), hint: "드리프트 기준선", icon: TimelineOutlined, to: "/drift" },
      ]} />

      {/* 최근 학습 + 빠른 작업 */}
      <Box sx={{ display: "grid", gap: 3, gridTemplateColumns: { xs: "1fr", md: "2fr 1fr" }, mb: 3 }}>
        <Card variant="outlined" sx={{ borderRadius: 3 }}>
          <CardContent>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1.5 }}>
              <Typography variant="h3">최근 학습 실행</Typography>
              <Button size="small" component={RouterLink} to="/pipe">전체 보기</Button>
            </Stack>
            {!runsApi.data ? <LinearProgress /> : recent.length === 0 ? (
              <Typography variant="body2" color="text.secondary" sx={{ py: 3, textAlign: "center" }}>
                아직 학습 실행이 없습니다. <RouterLink to="/pipe">첫 파이프라인을 실행</RouterLink>해 보세요.
              </Typography>
            ) : (
              <Stack divider={<Divider />}>
                {recent.map((r) => {
                  const st = activeStage(r); const metric = metricSummary(r);
                  return (
                    <Box key={r.id} component={RouterLink} to="/pipe"
                      sx={{ display: "block", textDecoration: "none", color: "inherit", py: 1.25, px: 1, mx: -1, borderRadius: 1, "&:hover": { bgcolor: "action.hover" } }}>
                      <Stack direction="row" alignItems="center" spacing={1.5}>
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Stack direction="row" spacing={1} alignItems="center">
                            <Typography variant="body2" fontWeight={600} noWrap>{r.name}</Typography>
                            {r.artifacts.served_name && <Chip size="small" variant="outlined" label={r.artifacts.served_name} sx={{ height: 18, fontSize: 11 }} />}
                          </Stack>
                          <Typography variant="caption" color="text.secondary" noWrap sx={{ display: "block" }}>
                            {st ? `${st.title} · ${st.status}` : "대기"}{metric ? ` · ${metric}` : ""}
                          </Typography>
                          <LinearProgress variant="determinate" value={progressOf(r)} sx={{ mt: 0.75, height: 4, borderRadius: 2 }}
                            color={r.status === "failed" ? "error" : r.status === "succeeded" ? "success" : "primary"} />
                        </Box>
                        <Chip size="small" color={RUN_COLOR[r.status] ?? "default"} variant="filled" label={RUN_LABEL[r.status] ?? r.status} />
                      </Stack>
                    </Box>
                  );
                })}
              </Stack>
            )}
          </CardContent>
        </Card>

        <Card variant="outlined" sx={{ borderRadius: 3 }}>
          <CardContent>
            <Typography variant="h3" sx={{ mb: 1.5 }}>빠른 작업</Typography>
            <Stack spacing={1}>
              {ACTIONS.map((a) => {
                const Icon = a.icon;
                return (
                  <CardActionArea key={a.path} component={RouterLink} to={a.path} sx={{ borderRadius: 2, p: 1.25, border: "1px solid", borderColor: "divider" }}>
                    <Stack direction="row" spacing={1.5} alignItems="center">
                      <Box sx={{ display: "grid", placeItems: "center", width: 34, height: 34, borderRadius: 1.5, bgcolor: "action.hover", color: "primary.main" }}>
                        <Icon fontSize="small" />
                      </Box>
                      <Box><Typography variant="body2" fontWeight={700}>{a.label}</Typography>
                        <Typography variant="caption" color="text.secondary">{a.desc}</Typography></Box>
                    </Stack>
                  </CardActionArea>
                );
              })}
            </Stack>
          </CardContent>
        </Card>
      </Box>

      {/* 도메인 헤드라인 카드 */}
      <Box sx={{ display: "grid", gap: 3, gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", lg: "repeat(3,1fr)" } }}>
        <DomainCard icon={GavelOutlined} title="승인 대기" to="/rel" accent="warning.main"
          headline={releases.data ? pending.length : "—"}>
          {pending.length === 0
            ? <Typography variant="caption" color="text.secondary">대기 중인 릴리즈 승인이 없습니다.</Typography>
            : <Stack spacing={0.5}>{pending.slice(0, 3).map((r) => (
                <Typography key={r.id} variant="caption" noWrap>{r.served_name || r.model_ref || r.id}</Typography>
              ))}</Stack>}
        </DomainCard>

        <DomainCard icon={CloudQueueOutlined} title="배포 · 서빙" to="/serving" accent="success.main"
          headline={models.data ? `${modelList.length} 모델` : "—"}>
          {modelList.length === 0
            ? <Typography variant="caption" color="text.secondary">등록된 서빙 모델이 없습니다.</Typography>
            : <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                {modelList.slice(0, 5).map((m, i) => <Chip key={i} size="small" variant="outlined" label={m.name ?? "model"} />)}
              </Stack>}
        </DomainCard>

        <DomainCard icon={MemoryOutlined} title="인프라 · GPU" to="/dash" accent="warning.main"
          headline={infra.data ? `GPU ${gpus.length} · 서비스 ${infra.data.services?.length ?? 0}` : "—"}>
          {gpus.length === 0
            ? <Typography variant="caption" color="text.secondary">검출된 GPU가 없습니다.</Typography>
            : <Stack spacing={0.75}>{gpus.slice(0, 2).map((g, i) => (
                <Box key={i}>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="caption" noWrap>{g.name ?? `GPU${i}`}</Typography>
                    <Typography variant="caption" color="text.secondary">{g.util != null ? `${g.util}%` : ""}</Typography>
                  </Stack>
                  <LinearProgress variant="determinate" value={Math.min(g.util ?? 0, 100)} sx={{ height: 5, borderRadius: 2 }} color="warning" />
                </Box>
              ))}</Stack>}
        </DomainCard>

        <DomainCard icon={TimelineOutlined} title="모니터링 · 알림" to="/drift" accent="error.main"
          headline={runsApi.data ? `주의 ${failedRuns.length + pending.length}건` : "—"}>
          {failedRuns.length === 0 && pending.length === 0
            ? <Typography variant="caption" color="text.secondary">이상 신호가 없습니다.</Typography>
            : <Stack spacing={0.5}>
                {failedRuns.map((r) => <Typography key={r.id} variant="caption" color="error.main" noWrap>실패: {r.name}</Typography>)}
                {pending.slice(0, 2).map((r) => <Typography key={r.id} variant="caption" color="warning.main" noWrap>승인 대기: {r.served_name || r.id}</Typography>)}
              </Stack>}
        </DomainCard>

        <DomainCard icon={VpnKeyOutlined} title="비용 · 거버넌스" to="/keys"
          headline={keys.data ? `${keys.data.length} 키` : "—"}>
          {(keys.data?.length ?? 0) === 0
            ? <Typography variant="caption" color="text.secondary">발급된 테넌트 키가 없습니다.</Typography>
            : <Stack spacing={0.5}>{(keys.data ?? []).slice(0, 3).map((k, i) => (
                <Typography key={i} variant="caption" noWrap>
                  {k.tenant || k.name || k.id}{k.budget_usd != null ? ` · $${(k.spent_usd ?? 0).toFixed(2)}/$${k.budget_usd}` : ""}
                </Typography>
              ))}</Stack>}
        </DomainCard>

        <DomainCard icon={TuneOutlined} title="HPO · 데이터" to="/hpo"
          headline={hpo.data ? `HPO ${hpo.data.length} · 데이터셋 ${datasets.data?.length ?? 0}` : "—"}>
          <Typography variant="caption" color="text.secondary">
            파라미터 탐색 스터디와 버전 고정 데이터셋 현황
          </Typography>
        </DomainCard>
      </Box>
    </>
  );
}

import { useCallback, useMemo, useState } from "react";
import {
  Box, Button, Card, CardContent, Chip, Divider, MenuItem, Stack, Table,
  TableBody, TableCell, TableHead, TableRow, TextField, Typography,
  IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import HourglassEmptyIcon from "@mui/icons-material/HourglassEmpty";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import type { SvgIconComponent } from "@mui/icons-material";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { useAutoRefresh } from "../hooks/useAutoRefresh";
import { api, ApiError } from "../api";
import { koStatus, stageColor, STAGE_KO } from "../lib/stages";

interface PipelineStage {
  name: string;
  title: string;
  status: string;
  detail: string;
}

interface PipelineRun {
  id: string;
  name: string;
  status: string;
  stages: PipelineStage[];
  release_id?: string | null;
  artifacts: Record<string, string>;
  loss_history: Record<string, number>[];
  created_at?: number | null;
}

const SFT_SAMPLE = `{"text": "환불 절차를 알려주세요.", "response": "주문 내역에서 환불 신청 후 3영업일 내 처리됩니다."}
{"text": "배송은 얼마나 걸리나요?", "response": "결제 완료 후 평균 2~3일 소요됩니다."}`;

const DPO_SAMPLE = `{"prompt": "환불 절차를 알려주세요.", "chosen": "주문 내역에서 환불 신청 후 3영업일 내 처리됩니다.", "rejected": "몰라요."}`;

function parseJsonl(txt: string): Record<string, unknown>[] {
  return txt
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l, i) => {
      try {
        return JSON.parse(l) as Record<string, unknown>;
      } catch {
        throw new Error(`${i + 1}번째 줄 JSON 파싱 실패`);
      }
    });
}

/** 상태 라벨을 팔레트 색으로 표시. */
function StatusLabel({ status }: { status: string }) {
  return (
    <Typography component="span" variant="body2" fontWeight={600} sx={{ color: stageColor(status) }}>
      {koStatus(status)}
    </Typography>
  );
}

/** 상태별 아이콘: 색뿐 아니라 모양으로도 상태를 구분해 마우스오버 없이 읽힌다. */
function stageIcon(status: string): SvgIconComponent {
  switch (status) {
    case "succeeded": return CheckCircleIcon;
    case "failed": return ErrorIcon;
    case "running":
    case "post-running": return HourglassEmptyIcon;
    case "waiting": return PauseCircleIcon;
    default: return RadioButtonUncheckedIcon; // pending, skipped 등
  }
}

/** 단계 진행 상황을 상태 아이콘들로 요약. 색+아이콘이라 hover 없이도 상태를 알 수 있다. */
function StageDots({ stages }: { stages: PipelineStage[] }) {
  return (
    <Stack direction="row" spacing={0.25} sx={{ flexWrap: "wrap", gap: 0.25 }}>
      {stages.map((s) => {
        const Icon = stageIcon(s.status);
        return (
          <Tooltip key={s.name} title={`${STAGE_KO[s.name] ?? s.title}: ${koStatus(s.status)}`}>
            <Icon
              aria-label={`${STAGE_KO[s.name] ?? s.title}: ${koStatus(s.status)}`}
              sx={{ fontSize: 16, color: stageColor(s.status) }}
            />
          </Tooltip>
        );
      })}
    </Stack>
  );
}

const RUNNING = ["running", "post-running"];

export default function Pipeline() {
  const { data, loading, error, reload } = useApi<PipelineRun[]>("/api/pipeline/runs");
  const runs = data ?? [];

  // 실행 폼 상태
  const [method, setMethod] = useState<"sft" | "dpo">("sft");
  const [pet, setPet] = useState<"lora" | "dora">("lora");
  const [servedName, setServedName] = useState("hcx-seed-tuned");
  const [maxSteps, setMaxSteps] = useState("30");
  const [dataText, setDataText] = useState(SFT_SAMPLE);
  const [runBusy, setRunBusy] = useState(false);
  const [runErr, setRunErr] = useState<string | null>(null);

  // 상세 조회 상태
  const [detail, setDetail] = useState<PipelineRun | null>(null);
  const [detailErr, setDetailErr] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState(false);

  const switchMethod = (m: "sft" | "dpo") => {
    setMethod(m);
    if (dataText === SFT_SAMPLE || dataText === DPO_SAMPLE || !dataText.trim()) {
      setDataText(m === "dpo" ? DPO_SAMPLE : SFT_SAMPLE);
    }
  };

  const openDetail = useCallback(async (runId: string) => {
    setDetailErr(null);
    try {
      const r = await api<PipelineRun>(`/api/pipeline/runs/${runId}`);
      setDetail(r);
    } catch (e) {
      setDetail(null);
      setDetailErr(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    }
  }, []);

  // 진행 중인 상세는 자동 새로고침(2초 간격).
  const detailRunning = detail != null && RUNNING.includes(detail.status);
  const detailId = detail?.id;
  useAutoRefresh(
    () => {
      if (detailId) {
        void openDetail(detailId);
        void reload();
      }
    },
    2000,
    detailRunning,
  );

  const startRun = async () => {
    setRunBusy(true);
    setRunErr(null);
    try {
      const rows = parseJsonl(dataText);
      const body: Record<string, unknown> = {
        name: "console-e2e",
        method,
        pet,
        served_name: servedName,
        train_max_steps: Number(maxSteps),
      };
      if (method === "dpo") body.preference = rows;
      else body.labeled = rows;
      const out = await api<PipelineRun>("/api/pipeline/run", { method: "POST", body });
      setDetail(out);
      setDetailErr(null);
      await reload();
    } catch (e) {
      setRunErr(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    } finally {
      setRunBusy(false);
    }
  };

  const resume = async (run: PipelineRun) => {
    setActionBusy(true);
    setDetailErr(null);
    try {
      // 승인 게이트: release 요청이 있으면 먼저 승인한 뒤 재개.
      if (run.release_id) {
        await api(`/api/releases/${run.release_id}/approve`, { method: "POST", body: { approver: "console" } });
      }
      const out = await api<PipelineRun>(`/api/pipeline/runs/${run.id}/resume`, { method: "POST" });
      setDetail(out);
      await reload();
    } catch (e) {
      setDetailErr(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    } finally {
      setActionBusy(false);
    }
  };

  const artifacts = useMemo(() => Object.entries(detail?.artifacts ?? {}), [detail]);

  return (
    <>
      <PageHeader
        title="파이프라인(직접 실행)"
        subtitle="데이터 적재·품질·학습·병합·평가·승인·배포를 한 번에 실행하고 진행 상황을 조회합니다."
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>파이프라인 실행</Typography>
          <Stack direction="row" spacing={2} sx={{ mb: 2, flexWrap: "wrap", gap: 2 }}>
            <TextField
              select label="데이터 유형" size="small" value={method}
              onChange={(e) => switchMethod(e.target.value as "sft" | "dpo")}
              sx={{ minWidth: 200 }}
            >
              <MenuItem value="sft">Instruction (SFT)</MenuItem>
              <MenuItem value="dpo">Preference (DPO)</MenuItem>
            </TextField>
            <TextField
              select label="PET" size="small" value={pet}
              onChange={(e) => setPet(e.target.value as "lora" | "dora")}
              sx={{ minWidth: 140 }}
            >
              <MenuItem value="lora">LoRA</MenuItem>
              <MenuItem value="dora">DoRA</MenuItem>
            </TextField>
            <TextField
              label="max_steps" size="small" value={maxSteps}
              onChange={(e) => setMaxSteps(e.target.value)}
              sx={{ width: 120 }}
            />
            <TextField
              label="배포 모델명" size="small" value={servedName}
              onChange={(e) => setServedName(e.target.value)}
              sx={{ minWidth: 200 }}
            />
          </Stack>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            학습 데이터 (<code>{method === "dpo" ? "{ prompt, chosen, rejected }" : "{ text, response }"}</code> JSONL · 한 줄에 하나)
          </Typography>
          <TextField
            multiline minRows={5} fullWidth value={dataText}
            onChange={(e) => setDataText(e.target.value)}
            sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
          />
          <Box sx={{ mt: 2 }}>
            <Button variant="contained" onClick={startRun} disabled={runBusy}>
              {runBusy ? "실행 중…" : "실행"}
            </Button>
          </Box>
          {runErr && <ErrorView message={runErr} />}
        </CardContent>
      </Card>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>실행 목록</Typography>
          {loading && !data ? (
            <Loading />
          ) : error ? (
            <ErrorView message={error} />
          ) : runs.length === 0 ? (
            <EmptyView message="아직 실행된 파이프라인이 없습니다. 위에서 실행하세요." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>이름</TableCell>
                  <TableCell>상태</TableCell>
                  <TableCell>단계</TableCell>
                  <TableCell align="right">액션</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {runs.map((r) => (
                  <TableRow key={r.id} hover selected={detail?.id === r.id}>
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{r.id}</TableCell>
                    <TableCell>{r.name}</TableCell>
                    <TableCell><StatusLabel status={r.status} /></TableCell>
                    <TableCell><StageDots stages={r.stages} /></TableCell>
                    <TableCell align="right">
                      <Stack direction="row" spacing={1} justifyContent="flex-end">
                        {r.status === "waiting" && (
                          <Button size="small" color="success" variant="contained"
                            disabled={actionBusy} onClick={() => resume(r)}>
                            재개(승인)
                          </Button>
                        )}
                        <Button size="small" variant="outlined" onClick={() => openDetail(r.id)}>
                          상세
                        </Button>
                      </Stack>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {(detail || detailErr) && (
        <Card>
          <CardContent>
            {detailErr && <ErrorView message={detailErr} />}
            {detail && (
              <>
                <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 1, flexWrap: "wrap" }}>
                  <Typography variant="h3" sx={{ fontFamily: "ui-monospace, monospace" }}>{detail.id}</Typography>
                  <StatusLabel status={detail.status} />
                  {detailRunning && <Chip size="small" variant="outlined" label="자동 새로고침" />}
                  <Box sx={{ flexGrow: 1 }} />
                  {detail.status === "waiting" && (
                    <Button color="success" variant="contained" disabled={actionBusy} onClick={() => resume(detail)}>
                      {actionBusy ? "재개 중…" : "재개(승인)"}
                    </Button>
                  )}
                  <Tooltip title="상세 새로고침">
                    <IconButton onClick={() => openDetail(detail.id)} aria-label="상세 새로고침"><RefreshIcon /></IconButton>
                  </Tooltip>
                </Stack>

                <Table size="small" sx={{ mb: 2 }}>
                  <TableHead>
                    <TableRow>
                      <TableCell>단계</TableCell>
                      <TableCell>상태</TableCell>
                      <TableCell>상세</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {detail.stages.map((s) => (
                      <TableRow key={s.name}>
                        <TableCell>{STAGE_KO[s.name] ?? s.title}</TableCell>
                        <TableCell><StatusLabel status={s.status} /></TableCell>
                        <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12, color: "text.secondary" }}>
                          {s.detail}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>

                {detail.loss_history.length > 0 && (
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    학습 loss: {detail.loss_history.length} steps
                  </Typography>
                )}

                {artifacts.length > 0 && (
                  <>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="body2" fontWeight={600} sx={{ mb: 1 }}>산출물</Typography>
                    <Stack spacing={0.5}>
                      {artifacts.map(([k, v]) => (
                        <Typography key={k} variant="caption" sx={{ fontFamily: "ui-monospace, monospace" }}>
                          <Box component="span" fontWeight={600}>{k}</Box>: {v}
                        </Typography>
                      ))}
                    </Stack>
                  </>
                )}

                {detail.status === "succeeded" && detail.artifacts.served_name && (
                  <Typography variant="body2" color="success.main" sx={{ mt: 2 }}>
                    배포 완료 → 챗 플레이그라운드에서 {detail.artifacts.served_name} 모델을 호출해 보세요.
                  </Typography>
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}
    </>
  );
}

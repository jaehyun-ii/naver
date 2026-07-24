import { type ReactNode, useCallback, useMemo, useState } from "react";
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
import { AiregTrainLogSection } from "../components/AiregSections";
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

const SFT_SAMPLE = `{"text": "선급증서 유효기간이 지나면 어떻게 해야 하나요?", "response": "선급이 정지될 수 있으므로 즉시 선급기관에 재검사를 신청해 증서를 갱신해야 합니다."}
{"text": "정기검사(Special Survey)는 몇 년 주기인가요?", "response": "정기검사는 5년 주기로 시행하며, 선체·기관·의장 상태를 종합 확인합니다."}`;

const DPO_SAMPLE = `{"prompt": "선급증서 유효기간이 지나면?", "chosen": "선급이 정지될 수 있으므로 즉시 선급기관에 재검사를 신청해 갱신합니다.", "rejected": "그냥 둬도 됩니다."}`;

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

const pj = (s?: string): Record<string, unknown> | null => {
  if (!s) return null;
  try { return JSON.parse(s); } catch { return null; }
};

// 학습 loss 하강 곡선 — 외부 차트 의존 없이 인라인 SVG 스파크라인
function Sparkline({ points }: { points: number[] }) {
  const w = 300, h = 60, pad = 5;
  const min = Math.min(...points), max = Math.max(...points), rng = max - min || 1;
  const X = (i: number) => pad + (i / (points.length - 1)) * (w - 2 * pad);
  const Y = (v: number) => pad + (1 - (v - min) / rng) * (h - 2 * pad);
  const line = points.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(" ");
  const area = `${line} L${X(points.length - 1).toFixed(1)} ${h - pad} L${X(0).toFixed(1)} ${h - pad} Z`;
  return (
    <Box sx={{ color: "primary.main" }}>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none" role="img" aria-label="학습 loss 곡선">
        <path d={area} fill="currentColor" opacity={0.1} />
        <path d={line} fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinejoin="round" />
        <circle cx={X(points.length - 1)} cy={Y(points[points.length - 1])} r={2.6} fill="currentColor" />
      </svg>
    </Box>
  );
}

function Tile({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Box sx={{ border: 1, borderColor: "divider", borderRadius: 2, p: 1.5, minWidth: 0 }}>
      <Typography variant="caption" color="text.secondary"
        sx={{ textTransform: "uppercase", letterSpacing: ".04em", fontWeight: 600, display: "block", mb: 0.5 }}>
        {label}
      </Typography>
      {children}
    </Box>
  );
}

// 학습이 실제로 됐는지 한눈에: loss곡선 · trainable수(%) · ΔW norm · 어댑터 존재 · base→adapter 개선폭
function TrainingVerification({ detail }: { detail: PipelineRun }) {
  const stats = pj(detail.artifacts.train_stats);
  const baseM = pj(detail.artifacts.base_metrics);
  const adaM = pj(detail.artifacts.adapter_metrics);
  const loss = (detail.loss_history ?? []).map((p) => Number(p.loss)).filter((v) => !Number.isNaN(v));
  const hasAdapter = Boolean(detail.artifacts.adapter);
  if (!loss.length && !stats && !adaM && !hasAdapter) return null;

  const first = loss[0], last = loss[loss.length - 1];
  const mkeys = adaM ? Object.keys(adaM).filter((k) => typeof adaM[k] === "number") : [];
  const tp = stats?.trainable_params as number | undefined;

  return (
    <>
      <Divider sx={{ my: 2 }} />
      <Typography variant="body2" fontWeight={700} sx={{ mb: 1.5 }}>학습 검증</Typography>
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" }, gap: 1.5 }}>

        {loss.length > 1 && (
          <Tile label={`학습 loss · ${loss.length} steps`}>
            <Sparkline points={loss} />
            <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.5 }}>
              <Typography variant="caption" color="text.secondary">start {first.toFixed(3)}</Typography>
              <Typography variant="caption" sx={{ color: last <= first ? "success.main" : "error.main", fontWeight: 700 }}>
                end {last.toFixed(3)} {last <= first ? "↓" : "↑"}
              </Typography>
            </Stack>
          </Tile>
        )}

        <Tile label="LoRA 파라미터 · 가중치 변화">
          {tp != null ? (
            <>
              <Typography variant="h4" sx={{ fontFamily: "ui-monospace, monospace", lineHeight: 1.15 }}>
                {tp.toLocaleString()}
                <Typography component="span" variant="caption" color="text.secondary"> 학습 / {Number(stats?.total_params).toLocaleString()}</Typography>
              </Typography>
              <Box sx={{ height: 6, borderRadius: 3, bgcolor: "action.hover", mt: 0.75, overflow: "hidden" }}>
                <Box sx={{ height: "100%", width: `${Math.min(Number(stats?.trainable_pct) || 0, 100)}%`, bgcolor: "primary.main" }} />
              </Box>
              <Typography variant="caption" color="text.secondary">전체의 {String(stats?.trainable_pct)}% 만 학습 (동결된 원본은 그대로)</Typography>
            </>
          ) : <Typography variant="caption" color="text.secondary">trainable 정보 없음</Typography>}
          <Stack direction="row" spacing={1} sx={{ mt: 1 }} alignItems="center" flexWrap="wrap" useFlexGap>
            {stats?.adapter_delta_norm != null && (
              <Chip size="small" variant="outlined" label={`ΔW norm ${String(stats.adapter_delta_norm)}`} />
            )}
            <Chip size="small" color={hasAdapter ? "success" : "default"} variant={hasAdapter ? "filled" : "outlined"}
              label={hasAdapter ? "어댑터 산출 ✓" : "어댑터 없음"} />
          </Stack>
        </Tile>

        {mkeys.length > 0 && (
          <Box sx={{ gridColumn: { md: "1 / -1" } }}>
            <Tile label="base(학습 전) → adapter(학습 후) 지표">
              <Stack spacing={1.25} sx={{ mt: 0.5 }}>
                {mkeys.map((k) => {
                  const a = Number(adaM![k]); const b = Number(baseM?.[k] ?? 0); const d = a - b; const up = d >= 0;
                  return (
                    <Box key={k}>
                      <Stack direction="row" justifyContent="space-between" alignItems="baseline">
                        <Typography variant="body2" sx={{ fontFamily: "ui-monospace, monospace" }}>{k}</Typography>
                        <Stack direction="row" spacing={1} alignItems="baseline">
                          <Typography variant="caption" color="text.secondary">{b.toFixed(3)} → {a.toFixed(3)}</Typography>
                          <Chip size="small" color={up ? "success" : "error"} label={`${up ? "+" : ""}${d.toFixed(3)} ${up ? "↑" : "↓"}`} />
                        </Stack>
                      </Stack>
                      <Box sx={{ position: "relative", height: 7, borderRadius: 3, bgcolor: "action.hover", mt: 0.5, overflow: "hidden" }}>
                        <Box sx={{ position: "absolute", top: 0, bottom: 0, left: 0, width: `${Math.max(0, Math.min(b * 100, 100))}%`, bgcolor: "text.disabled", opacity: 0.45 }} />
                        <Box sx={{ position: "absolute", top: 0, bottom: 0, left: 0, width: `${Math.max(0, Math.min(a * 100, 100))}%`, bgcolor: up ? "success.main" : "error.main", opacity: 0.85 }} />
                      </Box>
                    </Box>
                  );
                })}
              </Stack>
            </Tile>
          </Box>
        )}
      </Box>
    </>
  );
}

interface HpoStudy {
  id: string;
  status: string;
  method?: string;
  best_params: Record<string, number | string> | null;
  best_value?: number | null;
}

interface DsManifest { name: string; kind: string; fingerprint: string; num_train: number; samples?: Record<string, unknown>[] }

// 데이터셋 대표 샘플 → 학습 폼 JSONL(방식별 포맷)로 정규화.
function sampleToLine(s: Record<string, unknown>, kind: string): object | null {
  if (kind === "preference") {
    return s.prompt ? { prompt: s.prompt, chosen: s.chosen, rejected: s.rejected } : null;
  }
  if (s.text !== undefined) return { text: s.text, response: s.response };
  const msgs = s.messages as { role?: string; content?: string }[] | undefined;
  if (Array.isArray(msgs)) {
    const u = msgs.find((m) => m.role === "user")?.content;
    const a = msgs.find((m) => m.role === "assistant")?.content;
    if (u && a) return { text: u, response: a };
  }
  return null;
}
function datasetToJsonl(d: DsManifest): string {
  return (d.samples ?? [])
    .map((s) => sampleToLine(s, d.kind))
    .filter((x): x is object => x != null)
    .map((o) => JSON.stringify(o))
    .join("\n");
}

export default function Pipeline() {
  const { data, loading, error, reload } = useApi<PipelineRun[]>("/api/pipeline/runs");
  const runs = data ?? [];
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterQuery, setFilterQuery] = useState("");
  const filteredRuns = runs.filter((r) =>
    (filterStatus === "all" || r.status === filterStatus) &&
    (!filterQuery ||
      r.name.toLowerCase().includes(filterQuery.toLowerCase()) ||
      r.id.toLowerCase().includes(filterQuery.toLowerCase()) ||
      (r.artifacts.served_name || "").toLowerCase().includes(filterQuery.toLowerCase())));
  const { data: hpoData } = useApi<HpoStudy[]>("/api/tuning/hpo");
  const hpoStudies = (hpoData ?? []).filter((h) => h.status === "succeeded" && h.best_params);
  const datasets = useApi<DsManifest[]>("/api/data/datasets");

  // 실행 폼 상태
  const [method, setMethod] = useState<"sft" | "dpo">("sft");
  const [pet, setPet] = useState<"lora" | "dora">("lora");
  const [servedName, setServedName] = useState("hcx-seed-tuned");
  const [maxSteps, setMaxSteps] = useState("30");
  // 하이퍼파라미터 override (빈 값 = 학습기 기본값 사용)
  const [epochs, setEpochs] = useState("");
  const [lr, setLr] = useState("");
  const [loraR, setLoraR] = useState("");
  const [loraAlpha, setLoraAlpha] = useState("");
  const [gateThr, setGateThr] = useState("");
  const [hpoId, setHpoId] = useState("");
  const [selectedDs, setSelectedDs] = useState("");
  const [dataText, setDataText] = useState(SFT_SAMPLE);

  const pickDataset = (fp: string) => {
    setSelectedDs(fp);
    const d = datasets.data?.find((x) => x.fingerprint === fp);
    if (!d) return;
    setMethod(d.kind === "preference" ? "dpo" : "sft");
    const jsonl = datasetToJsonl(d);
    if (jsonl) setDataText(jsonl);
  };
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
      if (epochs.trim()) body.train_epochs = Number(epochs);
      if (lr.trim()) body.lr = Number(lr);
      if (loraR.trim()) body.lora_r = Number(loraR);
      if (loraAlpha.trim()) body.lora_alpha = Number(loraAlpha);
      if (gateThr.trim())
        body.gate_thresholds = { [method === "dpo" ? "preference_accuracy" : "answer_match"]: Number(gateThr) };
      if (hpoId) body.hpo_id = hpoId;
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

  // 학습 검증 패널이 시각화하는 원시 JSON은 산출물 목록에서 제외(중복 방지)
  const artifacts = useMemo(() => {
    const hidden = new Set(["train_stats", "base_metrics", "adapter_metrics", "eval_delta"]);
    return Object.entries(detail?.artifacts ?? {}).filter(([k]) => !hidden.has(k));
  }, [detail]);

  return (
    <>
      <PageHeader
        title="학습 파이프라인"
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
              sx={{ width: 110 }}
            />
            <TextField
              label="배포 모델명" size="small" value={servedName}
              onChange={(e) => setServedName(e.target.value)}
              sx={{ minWidth: 200 }}
            />
          </Stack>
          <Stack direction="row" spacing={2} sx={{ mb: 2, flexWrap: "wrap", gap: 2 }}>
            <TextField label="epochs" size="small" value={epochs} placeholder="1.0"
              onChange={(e) => setEpochs(e.target.value)} sx={{ width: 100 }} />
            <TextField label="learning rate" size="small" value={lr} placeholder="2e-4"
              onChange={(e) => setLr(e.target.value)} sx={{ width: 120 }} />
            <TextField label="lora_r" size="small" value={loraR} placeholder="16"
              onChange={(e) => setLoraR(e.target.value)} sx={{ width: 100 }} />
            <TextField label="lora_alpha" size="small" value={loraAlpha} placeholder="32"
              onChange={(e) => setLoraAlpha(e.target.value)} sx={{ width: 110 }} />
            <TextField label="게이트 임계값" size="small" value={gateThr} placeholder="0.7"
              onChange={(e) => setGateThr(e.target.value)} sx={{ width: 150 }}
              helperText={method === "dpo" ? "preference_accuracy" : "answer_match"} />
            <TextField
              select label="HPO 적용" size="small" value={hpoId}
              onChange={(e) => setHpoId(e.target.value)} sx={{ minWidth: 240 }}
              helperText={hpoStudies.length ? "완료된 HPO 스터디(선택 시 best_params 우선)" : "완료된 HPO 없음"}
            >
              <MenuItem value="">없음 (수동)</MenuItem>
              {hpoStudies.map((h) => (
                <MenuItem key={h.id} value={h.id}>
                  {h.id} · {h.method ?? "sft"}{h.best_value != null ? ` · score ${Number(h.best_value).toFixed(3)}` : ""}
                </MenuItem>
              ))}
            </TextField>
          </Stack>
          {hpoId ? (
            <Typography variant="caption" color="primary.main" sx={{ display: "block", mb: 1 }}>
              HPO 「{hpoId}」의 best_params 적용:{" "}
              {Object.entries(hpoStudies.find((h) => h.id === hpoId)?.best_params ?? {})
                .map(([k, v]) => `${k}=${v}`)
                .join(" · ")}
              {" "}— 위 수동 하이퍼파라미터는 무시됩니다.
            </Typography>
          ) : (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
              하이퍼파라미터는 비워 두면 학습기 기본값(lr 2e-4 · lora_r 16 · lora_alpha 32 · epochs 1.0)을 사용합니다.
            </Typography>
          )}
          <Stack direction="row" spacing={1.5} alignItems="flex-start" sx={{ mb: 1.5, flexWrap: "wrap", gap: 1 }}>
            <TextField select size="small" label="등록 데이터셋에서 불러오기" value={selectedDs}
              onChange={(e) => pickDataset(e.target.value)} sx={{ minWidth: 300 }}
              helperText={(datasets.data?.length ?? 0) === 0 ? "등록된 데이터셋 없음 — 데이터셋 페이지에서 빌드하세요" : "선택하면 대표 샘플이 아래에 로드됩니다"}>
              <MenuItem value="">직접 입력</MenuItem>
              {(datasets.data ?? []).map((d) => (
                <MenuItem key={d.fingerprint} value={d.fingerprint}>
                  {d.name} · {d.kind} · train {d.num_train} (샘플 {d.samples?.length ?? 0})
                </MenuItem>
              ))}
            </TextField>
          </Stack>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            학습 데이터 (<code>{method === "dpo" ? "{ prompt, chosen, rejected }" : "{ text, response }"}</code> JSONL · 한 줄에 하나)
            {selectedDs && <Typography component="span" variant="caption" color="primary.main" sx={{ ml: 1 }}>· 데이터셋에서 로드됨</Typography>}
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
          <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 1.5, flexWrap: "wrap", gap: 1 }}>
            <Typography variant="h3" sx={{ flexGrow: 1 }}>
              실행 목록 <Typography component="span" variant="caption" color="text.secondary">({filteredRuns.length}/{runs.length})</Typography>
            </Typography>
            <TextField select size="small" label="상태" value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)} sx={{ width: 130 }}>
              <MenuItem value="all">전체</MenuItem>
              <MenuItem value="running">진행 중</MenuItem>
              <MenuItem value="waiting">대기(승인)</MenuItem>
              <MenuItem value="succeeded">완료</MenuItem>
              <MenuItem value="failed">실패</MenuItem>
            </TextField>
            <TextField size="small" label="검색 (이름·ID·모델)" value={filterQuery} onChange={(e) => setFilterQuery(e.target.value)} sx={{ minWidth: 200 }} />
          </Stack>
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
                {filteredRuns.length === 0 && (
                  <TableRow><TableCell colSpan={5}>
                    <Typography variant="body2" color="text.secondary" sx={{ py: 1 }}>필터 조건에 맞는 실행이 없습니다.</Typography>
                  </TableCell></TableRow>
                )}
                {filteredRuns.map((r) => (
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

                <TrainingVerification detail={detail} />

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

      <Box sx={{ mt: 3 }}>
        <AiregTrainLogSection />
      </Box>
    </>
  );
}

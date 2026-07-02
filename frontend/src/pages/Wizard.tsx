import { useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, IconButton, MenuItem, Stack, Step, StepContent,
  StepLabel, Stepper, TextField, Tooltip, Typography,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView } from "../components/StateViews";
import { useAutoRefresh } from "../hooks/useAutoRefresh";
import { api, ApiError } from "../api";
import { STAGE_KO, koStatus, stageColor } from "../lib/stages";

interface PipelineStage {
  name: string;
  title: string;
  status: string;
  detail: string;
}

interface PipelineRun {
  id: string;
  name: string;
  status: string; // running | post-running | waiting | succeeded | failed
  stages: PipelineStage[];
  release_id?: string | null;
  artifacts: Record<string, string>;
  loss_history: { loss?: number }[];
  created_at?: number | null;
}

interface RunPipelineBody {
  name: string;
  method: "sft" | "dpo";
  served_name: string;
  train_max_steps: number;
  labeled?: Record<string, unknown>[];
  preference?: Record<string, unknown>[];
}

type Method = "sft" | "dpo";

const SAMPLE_QA = `{"text": "선급증서 유효기간이 지나면 어떻게 하나요?", "response": "즉시 선급기관에 재검사를 신청해 갱신합니다."}
{"text": "공장인수시험에 불합격하면?", "response": "부적합 항목을 시정조치 요구서로 발행하고 재시험합니다."}
{"text": "재료증명서가 없으면?", "response": "공급사에 제출을 요구하고 미제출 시 입고를 보류합니다."}`;

const SAMPLE_PREF = `{"prompt": "선급증서가 만료되면?", "chosen": "즉시 재검사를 신청해 갱신합니다.", "rejected": "그냥 두어도 됩니다."}
{"prompt": "재료증명서 누락 시?", "chosen": "제출을 요구하고 미제출 시 입고를 보류합니다.", "rejected": "일단 입고합니다."}`;

const PHASES = ["데이터 준비", "학습", "평가", "검토·승인", "배포"];

function errMsg(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

function parseJsonl(txt: string): Record<string, unknown>[] {
  const rows = txt
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l) as Record<string, unknown>);
  if (rows.length === 0) throw new Error("empty");
  return rows;
}

function stageLabel(s: PipelineStage): string {
  return STAGE_KO[s.name] ?? s.title ?? s.name;
}

/** 완료된 단계 수 = 현재 진행 중인 stepper 위치. */
function activeStepOf(run: PipelineRun): number {
  const i = run.stages.findIndex((s) => s.status !== "succeeded" && s.status !== "skipped");
  return i === -1 ? run.stages.length : i;
}

export default function Wizard() {
  const [kind, setKind] = useState<Method>("sft");
  const [name, setName] = useState("우리회사-모델");
  const [data, setData] = useState(SAMPLE_QA);
  const [run, setRun] = useState<PipelineRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const runId = run?.id;
  const active = !!run && (run.status === "running" || run.status === "post-running");

  // 실행 중에는 2초마다 상태를 폴링해 stepper 를 진행시킨다.
  useAutoRefresh(
    () => {
      if (!runId) return;
      api<PipelineRun>(`/api/pipeline/runs/${runId}`)
        .then(setRun)
        .catch(() => {});
    },
    2000,
    active,
  );

  // 새로고침: 현재 실행이 있으면 상세를 다시 불러온다(없으면 no-op).
  const reload = () => {
    if (!runId) return;
    api<PipelineRun>(`/api/pipeline/runs/${runId}`)
      .then(setRun)
      .catch(() => {});
  };

  const onKindChange = (m: Method) => {
    setKind(m);
    // 예시 데이터도 방식에 맞춰 바꿔 초심자가 그대로 체험할 수 있게 한다.
    setData(m === "dpo" ? SAMPLE_PREF : SAMPLE_QA);
  };

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      let rows: Record<string, unknown>[];
      try {
        rows = parseJsonl(data);
      } catch {
        throw new Error("입력 형식이 올바르지 않습니다. 예시처럼 한 줄에 하나씩 JSON을 입력하세요.");
      }
      const body: RunPipelineBody = {
        name,
        method: kind,
        served_name: name.replace(/[^a-zA-Z0-9_-]/g, "-") || "tuned",
        train_max_steps: 20,
        ...(kind === "dpo" ? { preference: rows } : { labeled: rows }),
      };
      const r = await api<PipelineRun>("/api/pipeline/run", { method: "POST", body });
      setRun(r);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const approve = async () => {
    if (!run) return;
    setBusy(true);
    setError(null);
    try {
      // 승인 게이트: 릴리스를 먼저 승인해야 resume 이 배포까지 진행된다.
      if (run.release_id) {
        await api(`/api/releases/${run.release_id}/approve`, {
          method: "POST",
          body: { approver: "console" },
        });
      }
      const r = await api<PipelineRun>(`/api/pipeline/runs/${run.id}/resume`, { method: "POST" });
      setRun(r);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setRun(null);
    setError(null);
  };

  const isWaiting = run?.status === "waiting";
  const isDone = run?.status === "succeeded";
  const isFailed = run?.status === "failed";

  return (
    <>
      <PageHeader
        title="가이드 실행"
        subtitle="데이터 → 학습 → 평가 → 승인 → 배포까지, 5단계로 나만의 AI 모델을 만들어 봅니다."
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>이렇게 진행됩니다</Typography>
          <Stepper alternativeLabel activeStep={-1} sx={{ mt: 1 }}>
            {PHASES.map((p) => (
              <Step key={p}>
                <StepLabel>{p}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </CardContent>
      </Card>

      {error && <ErrorView message={error} />}

      {!run ? (
        <Card>
          <CardContent>
            <Typography variant="h3" gutterBottom>1. 학습 데이터와 설정</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              AI에게 가르치고 싶은 내용을 한 줄에 하나씩 입력하세요. 아래 예시를 그대로 실행해 체험해도 됩니다.
            </Typography>

            <Stack spacing={2} sx={{ maxWidth: 520, mb: 2 }}>
              <TextField
                select
                label="학습 방식"
                value={kind}
                onChange={(e) => onKindChange(e.target.value as Method)}
                helperText="일반 질의응답이 대부분 적합합니다. 선호 비교는 좋은 답/나쁜 답을 비교해 정렬합니다."
              >
                <MenuItem value="sft">일반 질의응답 (권장)</MenuItem>
                <MenuItem value="dpo">선호 비교</MenuItem>
              </TextField>
              <TextField
                label="모델 이름"
                value={name}
                onChange={(e) => setName(e.target.value)}
                helperText="배포 후 이 이름으로 모델을 부릅니다."
              />
            </Stack>

            <TextField
              label="학습 데이터 (JSONL)"
              multiline
              minRows={6}
              fullWidth
              value={data}
              onChange={(e) => setData(e.target.value)}
              sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
              helperText={
                kind === "dpo"
                  ? "한 줄에 하나씩: { prompt, chosen, rejected }"
                  : "한 줄에 하나씩: { text, response }"
              }
            />

            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                size="large"
                onClick={start}
                disabled={busy || !name.trim() || !data.trim()}
              >
                {busy ? "시작하는 중…" : "시작"}
              </Button>
            </Box>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent>
            <Typography variant="h3" gutterBottom>2. 진행 상황</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              전체 상태: <Box component="span" sx={{ color: stageColor(run.status), fontWeight: 600 }}>{koStatus(run.status)}</Box>
              {" · "}실행 ID <Box component="span" sx={{ fontFamily: "ui-monospace, monospace" }}>{run.id}</Box>
            </Typography>

            <Stepper activeStep={activeStepOf(run)} orientation="vertical" sx={{ mb: 2 }}>
              {run.stages.map((s) => (
                <Step key={s.name} completed={s.status === "succeeded"}>
                  <StepLabel
                    error={s.status === "failed"}
                    optional={
                      <Typography variant="caption" sx={{ color: stageColor(s.status) }}>
                        {koStatus(s.status)}
                      </Typography>
                    }
                  >
                    {stageLabel(s)}
                  </StepLabel>
                  <StepContent>
                    {s.detail && (
                      <Typography variant="body2" color="text.secondary">{s.detail}</Typography>
                    )}
                  </StepContent>
                </Step>
              ))}
            </Stepper>

            {active && <Loading label="AI가 학습하고 평가하는 중입니다. 잠시 기다려 주세요…" />}

            {isWaiting && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                학습과 자동 평가를 마쳤습니다. 결과를 확인한 뒤 승인하면 서비스에 배포됩니다.
              </Alert>
            )}

            {isDone && (
              <Alert severity="success" sx={{ mb: 2 }}>
                배포 완료! 모델 <b>{run.artifacts?.served_name ?? run.name}</b> 이(가) 서비스에 올라갔습니다.
              </Alert>
            )}

            {isFailed && (
              <Alert severity="error" sx={{ mb: 2 }}>
                진행이 중단되었습니다. 데이터를 확인하고 다시 시도하세요.
              </Alert>
            )}

            <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: "wrap", gap: 1 }}>
              {isWaiting && (
                <Button variant="contained" onClick={approve} disabled={busy}>
                  {busy ? "승인 중…" : "승인하고 계속"}
                </Button>
              )}
              {(isDone || isFailed) && (
                <Button variant="outlined" onClick={reset} disabled={busy}>
                  처음부터 다시
                </Button>
              )}
            </Stack>
          </CardContent>
        </Card>
      )}
    </>
  );
}

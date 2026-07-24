import { useCallback, useEffect, useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, Chip, Divider, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
import RefreshIcon from "@mui/icons-material/Refresh";
import TuneOutlined from "@mui/icons-material/TuneOutlined";
import CheckCircleOutline from "@mui/icons-material/CheckCircleOutline";
import EmojiEventsOutlined from "@mui/icons-material/EmojiEventsOutlined";
import AccountTreeOutlined from "@mui/icons-material/AccountTreeOutlined";
import { AiregExperimentsSection } from "../components/AiregSections";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { SummaryTiles } from "../components/SummaryTiles";
import { ScatterPlot } from "../components/Charts";
import { useApi } from "../hooks/useApi";
import { useAutoRefresh } from "../hooks/useAutoRefresh";
import { api, ApiError } from "../api";
import { koStatus, stageColor } from "../lib/stages";

interface HpoTrial {
  trial: number;
  score: number;
  params: Record<string, number | string>;
}

interface HpoStudy {
  id: string;
  trials_n: number;
  status: string;
  created_at: number;
  trials: HpoTrial[];
  best_params: Record<string, number | string> | null;
  best_value: number | null;
  error?: string;
}

const SAMPLE_LABELED = `{"text": "정기검사 주기?", "response": "5년 주기로 선체·기관·의장을 종합 확인"}
{"text": "연차검사 시기?", "response": "매년, 기준일 전후 3개월 검사창 내"}`;

const SAMPLE_EVAL = `{"question": "정기검사 주기?", "expected": "5년 주기"}`;

function toApiError(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

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

function fmtNum(v: number | string | undefined): string {
  if (v === undefined || v === null) return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  return Number.isInteger(n) ? String(n) : n.toPrecision(4);
}

function fmtTime(ts: number): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString("ko-KR");
}

function StatusChip({ status }: { status: string }) {
  return <Chip size="small" variant="outlined" label={koStatus(status)} sx={{ color: stageColor(status), borderColor: stageColor(status) }} />;
}

function LaunchForm({ onStarted }: { onStarted: (id: string) => void }) {
  const [labeled, setLabeled] = useState(SAMPLE_LABELED);
  const [evalset, setEvalset] = useState(SAMPLE_EVAL);
  const [trials, setTrials] = useState("4");
  const [steps, setSteps] = useState("12");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const labeledRows = parseJsonl(labeled);
      const evalRows = evalset.trim() ? parseJsonl(evalset) : [];
      const nTrials = Number(trials);
      const nSteps = Number(steps);
      if (!Number.isFinite(nTrials) || nTrials < 1) throw new Error("trials 는 1 이상의 정수여야 합니다.");
      if (!Number.isFinite(nSteps) || nSteps < 1) throw new Error("steps 는 1 이상의 정수여야 합니다.");
      const rec = await api<HpoStudy>("/api/tuning/hpo", {
        method: "POST",
        body: { labeled: labeledRows, eval: evalRows, trials: nTrials, steps: nSteps },
      });
      onStarted(rec.id);
    } catch (e) {
      setError(toApiError(e));
    } finally {
      setBusy(false);
    }
  };

  const inputSx = { "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h3" gutterBottom>HPO 스터디 시작</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Optuna(TPE) 탐색. 각 trial = 짧은 SFT + 평가로 학습 컨테이너에서 실제 실행됩니다(분 단위·백그라운드).
        </Typography>

        <Typography variant="body2" fontWeight={600} sx={{ mb: 0.5 }}>
          학습 데이터 (JSONL · <code>{"{ text, response }"}</code>)
        </Typography>
        <TextField multiline minRows={4} fullWidth value={labeled} onChange={(e) => setLabeled(e.target.value)} sx={{ ...inputSx, mb: 2 }} />

        <Typography variant="body2" fontWeight={600} sx={{ mb: 0.5 }}>
          평가셋 (JSONL · <code>{"{ question, expected }"}</code> · 비우면 학습 데이터로 자동 구성)
        </Typography>
        <TextField multiline minRows={3} fullWidth value={evalset} onChange={(e) => setEvalset(e.target.value)} sx={{ ...inputSx, mb: 2 }} />

        <Stack direction="row" spacing={2} sx={{ mb: 2, flexWrap: "wrap", gap: 2 }}>
          <TextField label="trials (탐색 횟수)" type="number" size="small" value={trials}
            onChange={(e) => setTrials(e.target.value)} sx={{ width: 180 }} inputProps={{ min: 1 }} />
          <TextField label="steps (trial당 학습 스텝)" type="number" size="small" value={steps}
            onChange={(e) => setSteps(e.target.value)} sx={{ width: 220 }} inputProps={{ min: 1 }} />
        </Stack>

        <Button variant="contained" onClick={start} disabled={busy}>
          {busy ? "시작 중…" : "HPO 시작"}
        </Button>

        {error && <ErrorView message={error} />}
      </CardContent>
    </Card>
  );
}

function BestParams({ study }: { study: HpoStudy }) {
  if (!study.best_params) return null;
  return (
    <Alert severity="success" sx={{ mb: 2 }}>
      <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1, alignItems: "center" }}>
        <Typography variant="body2" fontWeight={600}>최적 파라미터</Typography>
        {Object.entries(study.best_params).map(([k, v]) => (
          <Chip key={k} size="small" variant="outlined" label={`${k}: ${fmtNum(v)}`} />
        ))}
        <Chip size="small" color="success" label={`score: ${fmtNum(study.best_value ?? undefined)}`} />
      </Stack>
    </Alert>
  );
}

function StudyDetail({ hpoId }: { hpoId: string }) {
  const [study, setStudy] = useState<HpoStudy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const r = await api<HpoStudy>(`/api/tuning/hpo/${hpoId}`);
      setStudy(r);
      setError(null);
    } catch (e) {
      setError(toApiError(e));
    } finally {
      setLoading(false);
    }
  }, [hpoId]);

  useEffect(() => {
    setLoading(true);
    setStudy(null);
    setError(null);
    load();
  }, [hpoId, load]);

  // running 인 동안 2초 간격 폴링
  useAutoRefresh(load, 2000, study?.status === "running");

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1, gap: 1, flexWrap: "wrap" }}>
          <Typography variant="h3" sx={{ fontFamily: "ui-monospace, monospace" }}>{hpoId}</Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            {study && <StatusChip status={study.status} />}
            <Tooltip title="새로고침">
              <IconButton onClick={() => load()} aria-label="상세 새로고침" size="small"><RefreshIcon fontSize="small" /></IconButton>
            </Tooltip>
          </Box>
        </Box>

        {loading && !study ? (
          <Loading />
        ) : error ? (
          <ErrorView message={error} />
        ) : !study ? (
          <EmptyView message="스터디 정보를 불러올 수 없습니다." />
        ) : (
          <>
            <Typography variant="caption" color="text.secondary">
              생성 {fmtTime(study.created_at)} · 목표 trials {study.trials_n} · 완료 {study.trials.length}
            </Typography>
            <Divider sx={{ my: 2 }} />

            {study.status === "failed" && study.error && (
              <ErrorView message={study.error} />
            )}

            <BestParams study={study} />

            {study.status === "running" && (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                탐색 진행 중… (2초 간격 자동 갱신)
              </Typography>
            )}

            {study.trials.length === 0 ? (
              <EmptyView message="아직 완료된 trial 이 없습니다." />
            ) : (
              <>
                {study.trials.length > 1 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" fontWeight={600} sx={{ mb: 0.5 }}>trial 점수 분포</Typography>
                    <ScatterPlot
                      points={study.trials.map((t) => ({
                        x: t.trial, y: t.score,
                        best: study.best_value !== null && study.best_value === t.score,
                        tooltip: [
                          `trial ${t.trial} · score ${t.score.toFixed(4)}`,
                          `lr ${t.params?.learning_rate ?? "—"}`,
                          `lora_r ${t.params?.lora_r ?? "—"} · alpha ${t.params?.lora_alpha ?? "—"}`,
                        ],
                      }))}
                      xlabel="trial" ylabel="score"
                      fmtX={(v) => String(Math.round(v))} fmtY={(v) => v.toFixed(3)}
                    />
                  </Box>
                )}
                <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>trial</TableCell>
                    <TableCell align="right">learning_rate</TableCell>
                    <TableCell align="right">lora_r</TableCell>
                    <TableCell align="right">lora_alpha</TableCell>
                    <TableCell align="right">score</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {study.trials.map((t, i) => {
                    const isBest = study.best_value !== null && study.best_value === t.score;
                    return (
                      <TableRow key={i} selected={isBest}>
                        <TableCell>
                          {t.trial}
                          {isBest && <Chip size="small" color="success" label="best" sx={{ ml: 1 }} />}
                        </TableCell>
                        <TableCell align="right">{fmtNum(t.params?.learning_rate)}</TableCell>
                        <TableCell align="right">{fmtNum(t.params?.lora_r)}</TableCell>
                        <TableCell align="right">{fmtNum(t.params?.lora_alpha)}</TableCell>
                        <TableCell align="right" sx={{ fontWeight: isBest ? 700 : 400 }}>{fmtNum(t.score)}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
                </Table>
              </>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default function Hpo() {
  const { data, loading, error, reload } = useApi<HpoStudy[]>("/api/tuning/hpo");
  const [selected, setSelected] = useState<string | null>(null);

  const onStarted = (id: string) => {
    setSelected(id);
    reload();
  };

  return (
    <>
      <PageHeader
        title="파라미터 최적화 (HPO)"
        subtitle="Optuna 기반 하이퍼파라미터 탐색 스터디 실행·조회"
        action={
          <Stack direction="row" spacing={1} alignItems="center">
            <Button variant="outlined" size="small" component={RouterLink} to="/pipe" startIcon={<AccountTreeOutlined />}>
              파이프라인에서 사용
            </Button>
            <Tooltip title="새로고침">
              <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
            </Tooltip>
          </Stack>
        }
      />

      {(data?.length ?? 0) > 0 && (
        <SummaryTiles stats={[
          { label: "스터디", value: data!.length, hint: "HPO 탐색", icon: TuneOutlined },
          { label: "완료", value: data!.filter((s) => s.status === "succeeded").length, icon: CheckCircleOutline, accent: "success.main" },
          { label: "최고 score", value: (() => { const v = data!.map((s) => s.best_value).filter((x): x is number => x != null); return v.length ? Math.max(...v).toFixed(3) : "—"; })(), hint: "reference F1", icon: EmojiEventsOutlined },
        ]} />
      )}

      <LaunchForm onStarted={onStarted} />

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>스터디 목록</Typography>
          {loading && !data ? (
            <Loading />
          ) : error ? (
            <ErrorView message={error} />
          ) : !data || data.length === 0 ? (
            <EmptyView message="아직 실행된 HPO 스터디가 없습니다. 위에서 새 스터디를 시작하세요." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell align="right">상태</TableCell>
                  <TableCell align="right">trials</TableCell>
                  <TableCell align="right">best score</TableCell>
                  <TableCell align="right">생성</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((s) => (
                  <TableRow
                    key={s.id}
                    hover
                    selected={s.id === selected}
                    onClick={() => setSelected(s.id)}
                    sx={{ cursor: "pointer" }}
                  >
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{s.id}</TableCell>
                    <TableCell align="right"><StatusChip status={s.status} /></TableCell>
                    <TableCell align="right">{s.trials.length}/{s.trials_n}</TableCell>
                    <TableCell align="right">{fmtNum(s.best_value ?? undefined)}</TableCell>
                    <TableCell align="right">{fmtTime(s.created_at)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {selected && <StudyDetail hpoId={selected} />}

      <Box sx={{ mt: 3 }}>
        <AiregExperimentsSection />
      </Box>
    </>
  );
}

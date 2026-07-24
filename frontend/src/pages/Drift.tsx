import { useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, Chip, FormControlLabel, LinearProgress,
  Stack, Switch, Table, TableBody, TableCell, TableHead, TableRow, TextField,
  Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import TimelineOutlined from "@mui/icons-material/TimelineOutlined";
import DatasetOutlined from "@mui/icons-material/DatasetOutlined";
import StraightenOutlined from "@mui/icons-material/Straighten";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { SummaryTiles } from "../components/SummaryTiles";
import { BarChart } from "../components/Charts";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

interface Baseline {
  name: string;
  n: number | null;
  avg_len: number;
  created_at?: number | null;
}

interface DriftReport {
  name: string;
  drift: boolean;
  scores: Record<string, number>;
  thresholds: Record<string, number>;
  n_reference: number;
  n_current: number;
  details: Record<string, unknown>;
}

interface AutoRetrainResult {
  drift: boolean;
  triggered: boolean;
  scores?: Record<string, number>;
  approved?: boolean;
  run_id?: string;
  served_name?: string;
  reason?: string;
}

type Verdict = "normal" | "warning" | "drift";
type VerdictColor = "success" | "warning" | "error";

const SAMPLE_REF = `선급증서 유효기간이 지나면 재검사를 신청한다.
공장인수시험 불합격 시 시정조치 요구서를 발행한다.
재료증명서가 없으면 입고를 보류한다.`;

const SAMPLE_CUR = `선급 규칙에 따라 검사 보고서를 제출한다.
도면 승인 의견서를 검토한다.`;

function lines(s: string): string[] {
  return s.split("\n").map((x) => x.trim()).filter(Boolean);
}

function toMessage(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

/** 드리프트 여부 + 점수/임계 근접도로 정상/경고/드리프트 3단계 판정. */
function verdictOf(rep: DriftReport): { verdict: Verdict; color: VerdictColor; label: string } {
  if (rep.drift) return { verdict: "drift", color: "error", label: "드리프트 감지" };
  const ratios = Object.entries(rep.scores).map(([k, v]) => {
    const th = rep.thresholds[k];
    return th ? v / th : 0;
  });
  const maxRatio = ratios.length ? Math.max(...ratios) : 0;
  if (maxRatio >= 0.8) return { verdict: "warning", color: "warning", label: "경고 (임계 근접)" };
  return { verdict: "normal", color: "success", label: "정상 (안정)" };
}

function BaselineForm({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("ship-qa");
  const [refTxt, setRefTxt] = useState(SAMPLE_REF);
  const [error, setError] = useState<string | null>(null);
  const [ok, setOk] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    setError(null);
    setOk(null);
    try {
      const r = await api<{ name: string; n: number; avg_len: number }>("/api/drift/baseline", {
        method: "POST",
        body: { name, texts: lines(refTxt) },
      });
      setOk(`기준선 '${r.name}' 등록 완료 · ${r.n}건 · 평균길이 ${r.avg_len}`);
      onDone();
    } catch (e) {
      setError(toMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h3" gutterBottom>기준선 등록</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          학습·기준 데이터 분포를 기준선으로 저장합니다. 한 줄에 한 텍스트를 입력하세요.
        </Typography>
        <TextField
          label="기준선명" size="small" value={name}
          onChange={(e) => setName(e.target.value)} sx={{ mb: 2, width: 240 }}
        />
        <TextField
          label="기준선 데이터(학습 분포)" multiline minRows={5} fullWidth value={refTxt}
          onChange={(e) => setRefTxt(e.target.value)}
          sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
        />
        <Box sx={{ mt: 2 }}>
          <Button variant="contained" onClick={submit} disabled={busy || !name.trim()}>
            {busy ? "등록 중…" : "기준선 등록"}
          </Button>
        </Box>
        {error && <ErrorView message={error} />}
        {ok && <Alert severity="success" sx={{ mt: 2 }}>{ok}</Alert>}
      </CardContent>
    </Card>
  );
}

function CheckPanel({ baselines }: { baselines: Baseline[] }) {
  const [name, setName] = useState("ship-qa");
  const [curTxt, setCurTxt] = useState(SAMPLE_CUR);
  const [rep, setRep] = useState<DriftReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const check = async () => {
    setBusy(true);
    setError(null);
    setRep(null);
    try {
      const r = await api<DriftReport>("/api/drift/check", {
        method: "POST",
        body: { name, texts: lines(curTxt) },
      });
      setRep(r);
    } catch (e) {
      setError(toMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const v = rep ? verdictOf(rep) : null;

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h3" gutterBottom>드리프트 검사</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          현재 라이브·평가 텍스트를 기준선과 비교해 PSI/JS 분포 변화를 점수화합니다.
        </Typography>
        <TextField
          label="비교 기준선명" size="small" value={name}
          onChange={(e) => setName(e.target.value)} sx={{ mb: 2, width: 240 }}
          helperText={baselines.length ? `등록된 기준선: ${baselines.map((b) => b.name).join(", ")}` : undefined}
        />
        <TextField
          label="현재 데이터(라이브/평가)" multiline minRows={4} fullWidth value={curTxt}
          onChange={(e) => setCurTxt(e.target.value)}
          sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
        />
        <Box sx={{ mt: 2 }}>
          <Button variant="contained" onClick={check} disabled={busy || !name.trim()}>
            {busy ? "검사 중…" : "드리프트 검사"}
          </Button>
        </Box>

        {error && <ErrorView message={error} />}

        {rep && v && (
          <Box sx={{ mt: 2 }}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2, flexWrap: "wrap", gap: 1 }}>
              <Chip label={v.label} color={v.color} size="small" />
              <Typography variant="body2" color="text.secondary">
                기준선 <b>{rep.name}</b> · ref={rep.n_reference} · cur={rep.n_current}
              </Typography>
            </Stack>
            <Box sx={{ mb: 2 }}>
              <BarChart
                data={Object.entries(rep.scores).map(([k, v]) => ({
                  label: k, value: v, threshold: rep.thresholds[k],
                }))}
                fmt={(v) => v.toFixed(3)}
              />
            </Box>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>지표</TableCell>
                  <TableCell align="right">점수</TableCell>
                  <TableCell align="right">임계</TableCell>
                  <TableCell align="right">근접도</TableCell>
                  <TableCell align="right">판정</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(rep.scores).map(([k, score]) => {
                  const th = rep.thresholds[k] ?? 0;
                  const over = th > 0 && score > th;
                  const ratio = th > 0 ? Math.min((score / th) * 100, 100) : 0;
                  return (
                    <TableRow key={k}>
                      <TableCell sx={{ fontFamily: "ui-monospace, monospace" }}>{k}</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 600 }}>{score.toFixed(3)}</TableCell>
                      <TableCell align="right" sx={{ color: "text.secondary" }}>{th.toFixed(3)}</TableCell>
                      <TableCell align="right" sx={{ width: 120 }}>
                        <LinearProgress
                          variant="determinate" value={ratio}
                          color={over ? "error" : ratio >= 80 ? "warning" : "success"}
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color={over ? "error.main" : "success.main"} fontWeight={600}>
                          {over ? "초과" : "정상"}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

function AutoRetrainPanel({ baselines }: { baselines: Baseline[] }) {
  const [baselineName, setBaselineName] = useState("ship-qa");
  const [curTxt, setCurTxt] = useState(SAMPLE_CUR);
  const [autoApprove, setAutoApprove] = useState(true);
  const [result, setResult] = useState<AutoRetrainResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await api<AutoRetrainResult>("/api/drift/auto-retrain", {
        method: "POST",
        body: {
          baseline_name: baselineName,
          current_texts: lines(curTxt),
          auto_approve: autoApprove,
        },
      });
      setResult(r);
    } catch (e) {
      setError(toMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h3" gutterBottom>자동 재학습 정책</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          폐루프: 드리프트 검사 후 감지 시 재학습 파이프라인을 트리거합니다. 게이트 통과분 자동 승인·배포를 켜거나 끌 수 있습니다.
        </Typography>
        <TextField
          label="비교 기준선명" size="small" value={baselineName}
          onChange={(e) => setBaselineName(e.target.value)} sx={{ mb: 2, width: 240 }}
          helperText={baselines.length ? `등록된 기준선: ${baselines.map((b) => b.name).join(", ")}` : undefined}
        />
        <TextField
          label="운영 데이터 표본" multiline minRows={4} fullWidth value={curTxt}
          onChange={(e) => setCurTxt(e.target.value)}
          sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
        />
        <FormControlLabel
          sx={{ mt: 1, display: "block" }}
          control={<Switch checked={autoApprove} onChange={(e) => setAutoApprove(e.target.checked)} />}
          label="게이트 통과 시 자동 승인·배포"
        />
        <Box sx={{ mt: 1 }}>
          <Button variant="contained" color="primary" onClick={run} disabled={busy || !baselineName.trim()}>
            {busy ? "실행 중…" : "자동 재학습 실행"}
          </Button>
        </Box>

        {error && <ErrorView message={error} />}

        {result && (
          <Box sx={{ mt: 2 }}>
            <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1 }}>
              <Chip
                label={result.drift ? "드리프트 감지" : "안정"}
                color={result.drift ? "error" : "success"}
                size="small"
              />
              <Chip
                label={result.triggered ? "재학습 트리거됨" : "재학습 미실행"}
                color={result.triggered ? "warning" : "default"}
                variant={result.triggered ? "filled" : "outlined"}
                size="small"
              />
              {result.approved !== undefined && (
                <Chip
                  label={result.approved ? "자동 승인·배포" : "승인 보류"}
                  color={result.approved ? "success" : "default"}
                  variant="outlined"
                  size="small"
                />
              )}
              {result.served_name && (
                <Chip label={`served: ${result.served_name}`} variant="outlined" size="small" />
              )}
              {result.run_id && (
                <Chip label={`run: ${result.run_id}`} variant="outlined" size="small" />
              )}
            </Stack>
            {result.reason && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>{result.reason}</Typography>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default function Drift() {
  const { data, loading, error, reload } = useApi<Baseline[]>("/api/drift/baselines");
  const baselines = data ?? [];

  return (
    <>
      <PageHeader
        title="드리프트 탐지"
        subtitle="입력 데이터 분포 변화 탐지 — 기준선 등록·검사·자동 재학습 (PSI/JS, GPU 불필요)"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      {(data?.length ?? 0) > 0 && (
        <SummaryTiles stats={[
          { label: "기준선", value: data!.length, hint: "등록된 baseline", icon: TimelineOutlined },
          { label: "총 표본", value: data!.reduce((a, b) => a + (b.n ?? 0), 0), hint: "reference 텍스트", icon: DatasetOutlined },
          { label: "평균 길이", value: Math.round(data!.reduce((a, b) => a + (b.avg_len ?? 0), 0) / Math.max(data!.length, 1)), hint: "문자", icon: StraightenOutlined },
        ]} />
      )}

      <BaselineForm onDone={() => reload()} />
      <CheckPanel baselines={baselines} />
      <AutoRetrainPanel baselines={baselines} />

      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>기준선 목록</Typography>
          {loading && !data ? (
            <Loading />
          ) : error ? (
            <ErrorView message={error} />
          ) : baselines.length === 0 ? (
            <EmptyView message="아직 등록된 기준선이 없습니다. 위에서 기준선을 등록하세요." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>이름</TableCell>
                  <TableCell align="right">건수</TableCell>
                  <TableCell align="right">평균길이</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {baselines.map((b) => (
                  <TableRow key={b.name}>
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace" }}>{b.name}</TableCell>
                    <TableCell align="right">{b.n ?? "—"}</TableCell>
                    <TableCell align="right">{b.avg_len}</TableCell>
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

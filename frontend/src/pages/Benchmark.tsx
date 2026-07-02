import { useEffect, useMemo, useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, Chip, MenuItem, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { BarChart, type BarDatum } from "../components/Charts";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

type BenchTask = "reference" | "preference";

interface BenchmarkSummary {
  name: string;
  task: BenchTask;
  num_cases: number;
  created_at?: number | null;
}

interface BenchmarkResult {
  id: string;
  benchmark: string;
  task: BenchTask;
  model: string;
  metrics: Record<string, number>;
  num_cases: number;
  errors: number;
  created_at?: number | null;
}

interface ResultsResponse {
  results: BenchmarkResult[];
  leaderboard: Record<string, BenchmarkResult[]>;
}

interface ModelsResponse {
  models: string[];
}

const REF_SAMPLE = `{"question":"선급증서 유효기간이 지나면?","expected":"즉시 선급기관에 재검사를 신청해 갱신합니다"}
{"question":"FAT 불합격 시 절차는?","expected":"부적합 항목을 시정조치 요구서로 발행하고 재시험합니다"}`;

const PREF_SAMPLE = `{"prompt":"선급증서 유효기간이 지나면?","chosen":"즉시 선급기관에 재검사를 신청해 갱신합니다","rejected":"그냥 둬도 됩니다"}
{"prompt":"FAT 불합격 시?","chosen":"시정조치 요구서를 발행하고 재시험합니다","rejected":"무시하고 출하합니다"}`;

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

function errMsg(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

function fmt(v: number | undefined): string {
  return typeof v === "number" ? v.toFixed(3) : "—";
}

function isPreference(task: BenchTask): boolean {
  return task === "preference";
}

function metricSummary(r: BenchmarkResult): string {
  if (isPreference(r.task)) {
    return `preference_accuracy=${fmt(r.metrics.preference_accuracy)} · preference_margin=${fmt(r.metrics.preference_margin)}`;
  }
  return `answer_match=${fmt(r.metrics.answer_match)} · reference_f1=${fmt(r.metrics.reference_f1)}`;
}

export default function Benchmark() {
  const list = useApi<BenchmarkSummary[]>("/api/benchmark");
  const results = useApi<ResultsResponse>("/api/benchmark/results");
  const modelsApi = useApi<ModelsResponse>("/api/models");
  const models = useMemo(() => modelsApi.data?.models ?? [], [modelsApi.data]);

  // 등록 폼 상태
  const [name, setName] = useState("조선-QA");
  const [task, setTask] = useState<BenchTask>("reference");
  const [cases, setCases] = useState(REF_SAMPLE);
  const [regBusy, setRegBusy] = useState(false);
  const [regError, setRegError] = useState<string | null>(null);
  const [regOk, setRegOk] = useState<string | null>(null);

  // 실행 상태
  const [model, setModel] = useState("");
  const [running, setRunning] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [last, setLast] = useState<BenchmarkResult | null>(null);

  useEffect(() => {
    if (!model && models.length) setModel(models[0]);
  }, [models, model]);

  const switchTask = (t: BenchTask) => {
    setTask(t);
    if (cases === REF_SAMPLE || cases === PREF_SAMPLE || !cases.trim()) {
      setCases(t === "preference" ? PREF_SAMPLE : REF_SAMPLE);
    }
  };

  const reloadAll = () => {
    list.reload();
    results.reload();
    modelsApi.reload();
  };

  const register = async () => {
    setRegBusy(true);
    setRegError(null);
    setRegOk(null);
    try {
      const rows = parseJsonl(cases);
      const r = await api<{ name: string; task: BenchTask; num_cases: number }>("/api/benchmark", {
        method: "POST",
        body: { name, task, cases: rows },
      });
      setRegOk(`등록 완료: ${r.name} (${r.task}) · ${r.num_cases}건`);
      list.reload();
    } catch (e) {
      setRegError(errMsg(e));
    } finally {
      setRegBusy(false);
    }
  };

  const run = async (benchName: string) => {
    if (!model) return;
    setRunning(benchName);
    setRunError(null);
    setLast(null);
    try {
      const r = await api<BenchmarkResult>("/api/benchmark/run", {
        method: "POST",
        body: { name: benchName, model },
      });
      setLast(r);
      results.reload();
    } catch (e) {
      setRunError(errMsg(e));
    } finally {
      setRunning(null);
    }
  };

  const benchmarks = list.data ?? [];
  const leaderboard = Object.entries(results.data?.leaderboard ?? {});

  return (
    <>
      <PageHeader
        title="벤치마크 평가"
        subtitle="명명·고정된 평가셋에 모델을 실행해 표준 메트릭을 산출하고 리더보드로 비교합니다."
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={reloadAll} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      <Stack direction={{ xs: "column", md: "row" }} spacing={3} sx={{ mb: 3 }} alignItems="stretch">
        {/* 등록 */}
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h3" gutterBottom>벤치마크 등록</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              <strong>reference</strong>=생성·정답 대비(answer_match) · <strong>preference</strong>=선호 정확도(chosen&gt;rejected).
              한 줄에 하나의 JSON 케이스를 입력하세요.
            </Typography>
            <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
              <TextField
                label="이름" size="small" value={name}
                onChange={(e) => setName(e.target.value)} sx={{ flex: 1 }}
              />
              <TextField
                label="유형" size="small" select value={task}
                onChange={(e) => switchTask(e.target.value as BenchTask)} sx={{ width: 200 }}
              >
                <MenuItem value="reference">reference (생성·정답)</MenuItem>
                <MenuItem value="preference">preference (선호)</MenuItem>
              </TextField>
            </Stack>
            <TextField
              multiline minRows={6} fullWidth value={cases}
              onChange={(e) => setCases(e.target.value)}
              placeholder={task === "preference"
                ? '{"prompt":..., "chosen":..., "rejected":...}'
                : '{"question":..., "expected":...}'}
              sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
            />
            <Box sx={{ mt: 2 }}>
              <Button variant="contained" onClick={register} disabled={regBusy}>
                {regBusy ? "등록 중…" : "등록"}
              </Button>
            </Box>
            {regError && <ErrorView message={regError} />}
            {regOk && <Alert severity="success" sx={{ mt: 2 }}>{regOk}</Alert>}
          </CardContent>
        </Card>

        {/* 실행 */}
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h3" gutterBottom>실행</Typography>
            <TextField
              label="모델" size="small" select value={model}
              onChange={(e) => setModel(e.target.value)}
              sx={{ minWidth: 220, mb: 2 }}
              helperText={models.length ? undefined : "등록된 모델이 없습니다"}
              disabled={!models.length}
            >
              {models.map((m) => (
                <MenuItem key={m} value={m}>{m}</MenuItem>
              ))}
            </TextField>

            {list.loading && !list.data ? (
              <Loading />
            ) : list.error ? (
              <ErrorView message={list.error} />
            ) : benchmarks.length === 0 ? (
              <EmptyView message="등록된 벤치마크가 없습니다. 왼쪽에서 먼저 등록하세요." />
            ) : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>벤치마크</TableCell>
                    <TableCell>유형</TableCell>
                    <TableCell align="right">케이스</TableCell>
                    <TableCell align="right"></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {benchmarks.map((b) => (
                    <TableRow key={b.name}>
                      <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{b.name}</TableCell>
                      <TableCell>
                        <Chip size="small" variant="outlined" label={b.task}
                          color={isPreference(b.task) ? "secondary" : "default"} />
                      </TableCell>
                      <TableCell align="right">{b.num_cases}</TableCell>
                      <TableCell align="right">
                        <Button
                          size="small" variant="outlined"
                          onClick={() => run(b.name)}
                          disabled={!model || running !== null}
                        >
                          {running === b.name ? "실행 중…" : "실행"}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}

            {runError && <ErrorView message={runError} />}
            {last && (
              <Alert severity={last.errors > 0 ? "warning" : "success"} sx={{ mt: 2 }}>
                최근 실행: <strong>{last.model}</strong> · {metricSummary(last)} · {last.num_cases}건
                {last.errors > 0 ? ` (에러 ${last.errors})` : ""}
              </Alert>
            )}
          </CardContent>
        </Card>
      </Stack>

      {/* 리더보드 */}
      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>리더보드</Typography>
          {results.loading && !results.data ? (
            <Loading />
          ) : results.error ? (
            <ErrorView message={results.error} />
          ) : leaderboard.length === 0 ? (
            <EmptyView message="아직 실행 결과가 없습니다. 위에서 벤치마크를 실행하세요." />
          ) : (
            <Stack spacing={3}>
              {leaderboard.map(([bench, rows]) => {
                const pref = isPreference(rows[0]?.task ?? "reference");
                const barData: BarDatum[] = rows.map((r) => ({
                  label: r.model,
                  value: pref ? r.metrics.preference_accuracy : r.metrics.answer_match,
                }));
                return (
                  <Box key={bench}>
                    <Typography variant="body2" fontWeight={600} sx={{ mb: 1 }}>
                      {bench}
                      <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                        ({pref ? "preference_accuracy" : "answer_match"} 순)
                      </Typography>
                    </Typography>
                    {barData.length > 0 && (
                      <BarChart data={barData} fmt={(v) => fmt(v)} />
                    )}
                    <Table size="small" sx={{ mt: 2 }}>
                      <TableHead>
                        <TableRow>
                          <TableCell align="right">#</TableCell>
                          <TableCell>모델</TableCell>
                          <TableCell align="right">{pref ? "preference_accuracy" : "answer_match"}</TableCell>
                          <TableCell align="right">{pref ? "preference_margin" : "reference_f1"}</TableCell>
                          <TableCell align="right">케이스</TableCell>
                          <TableCell align="right">에러</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {rows.map((r, i) => (
                          <TableRow key={r.id}>
                            <TableCell align="right">{i + 1}</TableCell>
                            <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{r.model}</TableCell>
                            <TableCell align="right">
                              <Typography component="span" variant="body2" fontWeight={600} color="text.primary">
                                {fmt(pref ? r.metrics.preference_accuracy : r.metrics.answer_match)}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              {fmt(pref ? r.metrics.preference_margin : r.metrics.reference_f1)}
                            </TableCell>
                            <TableCell align="right">{r.num_cases}</TableCell>
                            <TableCell align="right">{r.errors}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Box>
                );
              })}
            </Stack>
          )}
        </CardContent>
      </Card>
    </>
  );
}

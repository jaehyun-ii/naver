import { useCallback, useEffect, useState } from "react";
import {
  Alert, Box, Card, CardContent, Chip, LinearProgress, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, TextField, MenuItem, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

interface ServicesResp {
  available: boolean;
  services?: string[];
  error?: string;
  jaeger?: string;
}

interface TraceSummary {
  traceID: string;
  root: string;
  spans: number;
  duration_ms: number;
  error: boolean;
  total_tokens?: number | string | null;
  cost_usd?: number | string | null;
}

interface TracesResp {
  available: boolean;
  traces?: TraceSummary[];
  error?: string;
  jaeger?: string;
}

interface TraceSpan {
  op: string;
  duration_ms: number;
  tags: Record<string, unknown>;
}

interface TraceDetailResp {
  available: boolean;
  spans?: TraceSpan[];
  error?: string;
}

function toMsg(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms} ms`;
}

/** Jaeger 미도달 시 백엔드는 available:false 로 응답한다(200). */
function unavailableMessage(r: { error?: string; jaeger?: string }): string {
  const parts = ["Jaeger 조회 API에 도달하지 못했습니다."];
  if (r.jaeger) parts.push(`(${r.jaeger})`);
  if (r.error) parts.push(r.error);
  return parts.join(" ");
}

function SpanWaterfall({ spans }: { spans: TraceSpan[] }) {
  const max = Math.max(1, ...spans.map((s) => s.duration_ms));
  return (
    <Stack spacing={1}>
      {spans.map((s, i) => {
        const pct = Math.min(100, Math.max(2, (s.duration_ms / max) * 100));
        const hasErr = s.tags.error === true || s.tags.error === "true";
        return (
          <Box key={i} sx={{ pl: 1 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5, gap: 1 }}>
              <Typography
                variant="body2"
                fontWeight={600}
                sx={{ fontFamily: "ui-monospace, monospace", fontSize: 13, wordBreak: "break-all" }}
              >
                {s.op}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: "nowrap" }}>
                {fmtMs(s.duration_ms)}
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={pct}
              color={hasErr ? "error" : "primary"}
            />
            {Object.keys(s.tags).length > 0 && (
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 0.5, mt: 0.5 }}>
                {Object.entries(s.tags).map(([k, v]) => (
                  <Chip key={k} size="small" variant="outlined" label={`${k}: ${String(v)}`} />
                ))}
              </Stack>
            )}
          </Box>
        );
      })}
    </Stack>
  );
}

function TraceDetail({ traceId }: { traceId: string }) {
  const [data, setData] = useState<TraceDetailResp | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    setData(null);
    api<TraceDetailResp>(`/api/traces/${encodeURIComponent(traceId)}`)
      .then((r) => alive && setData(r))
      .catch((e: unknown) => alive && setError(toMsg(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [traceId]);

  if (loading) return <Loading label="스팬 불러오는 중…" />;
  if (error) return <ErrorView message={error} />;
  if (!data) return null;
  if (!data.available) return <ErrorView message={unavailableMessage(data)} />;
  if (!data.spans || data.spans.length === 0) return <EmptyView message="스팬이 없습니다." />;

  return (
    <Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        {data.spans.length}개 스팬 · 시작 순 정렬
      </Typography>
      <SpanWaterfall spans={data.spans} />
    </Box>
  );
}

export default function Traces() {
  const services = useApi<ServicesResp>("/api/traces/services");
  const [service, setService] = useState<string>("");
  const [tracesData, setTracesData] = useState<TracesResp | null>(null);
  const [tracesLoading, setTracesLoading] = useState(false);
  const [tracesError, setTracesError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const svcList = services.data?.available ? services.data.services ?? [] : [];

  // 서비스 목록 로드 후 첫 항목 자동 선택.
  useEffect(() => {
    if (!service && svcList.length > 0) setService(svcList[0]);
  }, [svcList, service]);

  const loadTraces = useCallback((svc: string) => {
    if (!svc) return;
    setTracesLoading(true);
    setTracesError(null);
    setSelected(null);
    api<TracesResp>(`/api/traces?service=${encodeURIComponent(svc)}&limit=20`)
      .then((r) => setTracesData(r))
      .catch((e: unknown) => setTracesError(toMsg(e)))
      .finally(() => setTracesLoading(false));
  }, []);

  useEffect(() => {
    if (service) loadTraces(service);
    else setTracesData(null);
  }, [service, loadTraces]);

  const refresh = () => {
    services.reload();
    if (service) loadTraces(service);
  };

  const traces = tracesData?.available ? tracesData.traces ?? [] : [];

  return (
    <>
      <PageHeader
        title="트레이스"
        subtitle="Jaeger 통합 뷰 — LLM 트레이스·스팬을 콘솔에서 조회(읽기전용)"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={refresh} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>서비스 선택</Typography>
          {services.loading && !services.data ? (
            <Loading />
          ) : services.error ? (
            <ErrorView message={services.error} />
          ) : services.data && !services.data.available ? (
            <Alert severity="warning">{unavailableMessage(services.data)}</Alert>
          ) : svcList.length === 0 ? (
            <EmptyView message="등록된 서비스가 없습니다 (아직 트레이스가 수집되지 않았을 수 있음)." />
          ) : (
            <TextField
              select
              size="small"
              label="서비스"
              value={service}
              onChange={(e) => setService(e.target.value)}
              sx={{ minWidth: 260 }}
            >
              {svcList.map((s) => (
                <MenuItem key={s} value={s}>{s}</MenuItem>
              ))}
            </TextField>
          )}
        </CardContent>
      </Card>

      {service && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h3" gutterBottom>최근 트레이스</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              지연 시간 내림차순 · 최대 20건. 행을 클릭하면 스팬을 확인합니다.
            </Typography>
            {tracesLoading && !tracesData ? (
              <Loading />
            ) : tracesError ? (
              <ErrorView message={tracesError} />
            ) : tracesData && !tracesData.available ? (
              <ErrorView message={unavailableMessage(tracesData)} />
            ) : traces.length === 0 ? (
              <EmptyView message="이 서비스에 대한 트레이스가 없습니다." />
            ) : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Trace ID</TableCell>
                    <TableCell>루트 오퍼레이션</TableCell>
                    <TableCell align="right">스팬</TableCell>
                    <TableCell align="right">지연</TableCell>
                    <TableCell align="right">토큰</TableCell>
                    <TableCell align="right">비용(USD)</TableCell>
                    <TableCell align="right">상태</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {traces.map((t) => (
                    <TableRow
                      key={t.traceID}
                      hover
                      selected={selected === t.traceID}
                      onClick={() => setSelected(t.traceID)}
                      sx={{ cursor: "pointer" }}
                    >
                      <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                        {t.traceID.slice(0, 12)}
                      </TableCell>
                      <TableCell>{t.root || "—"}</TableCell>
                      <TableCell align="right">{t.spans}</TableCell>
                      <TableCell align="right">{fmtMs(t.duration_ms)}</TableCell>
                      <TableCell align="right">{t.total_tokens ?? "—"}</TableCell>
                      <TableCell align="right">{t.cost_usd ?? "—"}</TableCell>
                      <TableCell align="right">
                        <Chip
                          size="small"
                          variant="outlined"
                          label={t.error ? "오류" : "정상"}
                          color={t.error ? "error" : "success"}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      )}

      {selected && (
        <Card>
          <CardContent>
            <Typography variant="h3" gutterBottom>스팬 워터폴</Typography>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mb: 2, fontFamily: "ui-monospace, monospace", fontSize: 12, wordBreak: "break-all" }}
            >
              trace: {selected}
            </Typography>
            <TraceDetail traceId={selected} />
          </CardContent>
        </Card>
      )}
    </>
  );
}

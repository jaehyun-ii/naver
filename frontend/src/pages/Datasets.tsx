import { useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, Chip, Collapse, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
import RefreshIcon from "@mui/icons-material/Refresh";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import DatasetOutlined from "@mui/icons-material/DatasetOutlined";
import LayersOutlined from "@mui/icons-material/LayersOutlined";
import ModelTrainingOutlined from "@mui/icons-material/ModelTraining";
import CompareArrowsOutlined from "@mui/icons-material/CompareArrows";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { SummaryTiles } from "../components/SummaryTiles";
import { Histogram } from "../components/Charts";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

interface DatasetManifest {
  name: string;
  kind: string;
  fingerprint: string;
  num_train: number;
  num_val: number;
  num_test: number;
  s3_uri?: string | null;
  samples?: Record<string, unknown>[];
}

interface DataQualityReport {
  suite: string;
  num_records: number;
  passed: boolean;
  failures: Record<string, string>;
  stats: Record<string, number>;
}

const SAMPLE = `{"id": "1", "text": "선급증서는 선박이 선급규칙에 적합함을 증명하며 정기검사 주기에 맞춰 유효성을 유지한다."}
{"id": "2", "text": "정기검사(Special Survey)는 5년 주기로 선체·기관·의장 상태를 종합 확인한다."}`;

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

function ValidatePanel() {
  const [text, setText] = useState(SAMPLE);
  const [report, setReport] = useState<DataQualityReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      const records = parseJsonl(text);
      const r = await api<DataQualityReport>("/api/data/validate", { method: "POST", body: { records } });
      setReport(r);
    } catch (e) {
      const msg = e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
      setError(msg);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h3" gutterBottom>JSONL 품질 검증</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          한 줄에 하나의 JSON 레코드(<code>{"{ id, text, ... }"}</code>)를 입력하세요.
        </Typography>
        <TextField
          multiline minRows={6} fullWidth value={text}
          onChange={(e) => setText(e.target.value)}
          sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
        />
        <Box sx={{ mt: 2 }}>
          <Button variant="contained" onClick={run} disabled={busy}>
            {busy ? "검증 중…" : "검증 실행"}
          </Button>
        </Box>

        {error && <ErrorView message={error} />}

        {report && (
          <Box sx={{ mt: 2 }}>
            <Alert severity={report.passed ? "success" : "error"} sx={{ mb: 2 }}>
              {report.passed ? "품질 게이트 통과" : "품질 게이트 실패"} · {report.num_records}건 · suite: {report.suite}
            </Alert>

            {Object.keys(report.failures).length > 0 && (
              <>
                <Typography variant="body2" fontWeight={600} sx={{ mb: 1 }}>위반 내역</Typography>
                <Table size="small" sx={{ mb: 2 }}>
                  <TableBody>
                    {Object.entries(report.failures).map(([rule, desc]) => (
                      <TableRow key={rule}>
                        <TableCell sx={{ fontWeight: 600, width: "30%" }}>{rule}</TableCell>
                        <TableCell>{desc}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </>
            )}

            {Object.keys(report.stats).length > 0 && (
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1 }}>
                {Object.entries(report.stats).map(([k, v]) => (
                  <Chip key={k} size="small" variant="outlined" label={`${k}: ${Number(v).toFixed(3)}`} />
                ))}
              </Stack>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

function SField({ label, value }: { label: string; value: unknown }) {
  return (
    <Box sx={{ mb: 0.75 }}>
      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5 }}>
        {label}
      </Typography>
      <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>{String(value ?? "")}</Typography>
    </Box>
  );
}

function SampleCard({ s, i }: { s: Record<string, unknown>; i: number }) {
  const msgs = s.messages as { role: string; content: string }[] | undefined;
  return (
    <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.5, mb: 1 }}>
      <Typography variant="caption" color="text.secondary">#{i + 1}</Typography>
      {msgs ? (
        msgs.map((m, j) => <SField key={j} label={m.role} value={m.content} />)
      ) : s.text !== undefined ? (
        <><SField label="text" value={s.text} /><SField label="response" value={s.response} /></>
      ) : s.prompt !== undefined ? (
        <>
          <SField label="prompt" value={s.prompt} />
          <SField label="chosen" value={s.chosen} />
          <SField label="rejected" value={s.rejected} />
        </>
      ) : (
        <Typography variant="body2" sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12, whiteSpace: "pre-wrap" }}>
          {JSON.stringify(s, null, 2)}
        </Typography>
      )}
    </Box>
  );
}

function DatasetRow({ d }: { d: DatasetManifest }) {
  const [open, setOpen] = useState(false);
  const samples = d.samples ?? [];
  const has = samples.length > 0;
  return (
    <>
      <TableRow
        hover
        sx={{ cursor: has ? "pointer" : "default", "& > *": { borderBottom: "unset" } }}
        onClick={() => has && setOpen((o) => !o)}
      >
        <TableCell sx={{ width: 44 }}>
          {has && <IconButton size="small" aria-label="상세">{open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}</IconButton>}
        </TableCell>
        <TableCell>{d.name}</TableCell>
        <TableCell><Chip size="small" label={d.kind} variant="outlined" /></TableCell>
        <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{d.fingerprint.slice(0, 10)}</TableCell>
        <TableCell align="right">{d.num_train}</TableCell>
        <TableCell align="right">{d.num_val}</TableCell>
        <TableCell align="right">{d.num_test}</TableCell>
      </TableRow>
      <TableRow>
        <TableCell colSpan={7} sx={{ py: 0, borderBottom: open ? undefined : "none" }}>
          <Collapse in={open} unmountOnExit>
            <Box sx={{ py: 2 }}>
              <Typography variant="body2" fontWeight={700} sx={{ mb: 1 }}>
                데이터 미리보기 · {samples.length}건{d.num_train ? ` (train ${d.num_train}건 중)` : ""}
              </Typography>
              {samples.map((s, i) => <SampleCard key={i} s={s} i={i} />)}
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

export default function Datasets() {
  const { data, loading, error, reload } = useApi<DatasetManifest[]>("/api/data/datasets");

  return (
    <>
      <PageHeader
        title="데이터셋"
        subtitle="버전 고정된 학습 데이터셋 목록과 품질 검증"
        action={
          <Stack direction="row" spacing={1} alignItems="center">
            <Button variant="outlined" size="small" component={RouterLink} to="/pipe" startIcon={<ModelTrainingOutlined />}>
              학습 파이프라인으로
            </Button>
            <Tooltip title="새로고침">
              <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
            </Tooltip>
          </Stack>
        }
      />

      {(data?.length ?? 0) > 0 && (
        <SummaryTiles stats={[
          { label: "데이터셋", value: data!.length, hint: "버전 고정", icon: DatasetOutlined },
          { label: "train 합계", value: data!.reduce((a, d) => a + d.num_train, 0), hint: "학습 샘플", icon: LayersOutlined },
          { label: "SFT", value: data!.filter((d) => d.kind === "sft").length, icon: ModelTrainingOutlined, accent: "success.main" },
          { label: "Preference", value: data!.filter((d) => d.kind === "preference").length, icon: CompareArrowsOutlined },
        ]} />
      )}

      {(() => {
        const lens = (data ?? []).flatMap((d) => (d.samples ?? []).map((s) => {
          const v = (s.text ?? s.prompt ?? s.content) as unknown;
          if (typeof v === "string") return v.length;
          const msgs = s.messages as { content?: string }[] | undefined;
          if (Array.isArray(msgs)) return msgs.map((m) => m.content ?? "").join(" ").length;
          return JSON.stringify(s).length;
        }));
        return lens.length > 1 ? (
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h3" gutterBottom>샘플 길이 분포</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                전체 데이터셋 대표 샘플 {lens.length}건의 문자 길이 분포 — 편향·이상치 파악용.
              </Typography>
              <Histogram values={lens} unit="자" />
            </CardContent>
          </Card>
        ) : null;
      })()}

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>데이터셋 목록</Typography>
          {loading && !data ? (
            <Loading />
          ) : error ? (
            <ErrorView message={error} />
          ) : !data || data.length === 0 ? (
            <EmptyView message="아직 등록된 데이터셋이 없습니다. 파이프라인을 실행하거나 아래에서 검증 후 빌드하세요." />
          ) : (
            <>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                행을 클릭하면 실제 데이터 샘플을 볼 수 있습니다.
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell />
                    <TableCell>이름</TableCell>
                    <TableCell>종류</TableCell>
                    <TableCell>지문</TableCell>
                    <TableCell align="right">train</TableCell>
                    <TableCell align="right">val</TableCell>
                    <TableCell align="right">test</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.map((d) => <DatasetRow key={d.fingerprint} d={d} />)}
                </TableBody>
              </Table>
            </>
          )}
        </CardContent>
      </Card>

      <ValidatePanel />
    </>
  );
}

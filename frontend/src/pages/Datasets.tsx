import { useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, Chip, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
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
}

interface DataQualityReport {
  suite: string;
  num_records: number;
  passed: boolean;
  failures: Record<string, string>;
  stats: Record<string, number>;
}

const SAMPLE = `{"id": "1", "text": "안녕하세요, 오늘 날씨 어때요?"}
{"id": "2", "text": "환불 절차를 알려주세요."}`;

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

export default function Datasets() {
  const { data, loading, error, reload } = useApi<DatasetManifest[]>("/api/data/datasets");

  return (
    <>
      <PageHeader
        title="데이터셋"
        subtitle="버전 고정된 학습 데이터셋 목록과 품질 검증"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

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
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>이름</TableCell>
                  <TableCell>종류</TableCell>
                  <TableCell>지문</TableCell>
                  <TableCell align="right">train</TableCell>
                  <TableCell align="right">val</TableCell>
                  <TableCell align="right">test</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((d) => (
                  <TableRow key={d.fingerprint}>
                    <TableCell>{d.name}</TableCell>
                    <TableCell><Chip size="small" label={d.kind} variant="outlined" /></TableCell>
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                      {d.fingerprint.slice(0, 10)}
                    </TableCell>
                    <TableCell align="right">{d.num_train}</TableCell>
                    <TableCell align="right">{d.num_val}</TableCell>
                    <TableCell align="right">{d.num_test}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <ValidatePanel />
    </>
  );
}

import { useState } from "react";
import {
  Box, Button, Card, CardContent, Chip, Divider, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import MenuBookOutlined from "@mui/icons-material/MenuBookOutlined";
import LayersOutlined from "@mui/icons-material/LayersOutlined";
import SearchOutlined from "@mui/icons-material/Search";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { SummaryTiles } from "../components/SummaryTiles";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

interface RagStats {
  documents: number;
  embedder: string;
  dim: number;
  backend: string;
  collection: string;
  top_k: number;
}

interface IngestResult {
  ingested: number;
  total: number;
}

interface RagHit {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, unknown>;
}

interface QueryResult {
  hits: RagHit[];
}

interface RagMessage {
  role: string;
  content: string;
}

interface AugmentUsed {
  id: string;
  score: number;
}

interface AugmentResult {
  messages: RagMessage[];
  used: AugmentUsed[];
}

const SAMPLE_DOCS = `선급증서 유효기간이 지나면 즉시 선급기관에 재검사를 신청해 갱신한다.
FAT 불합격 시 부적합 항목을 시정조치 요구서로 발행하고 재시험한다.
도크 진수 전 선급기관 입회 검사를 통과해야 한다.`;

const SAMPLE_QUERY = "선급증서 만료되면 어떻게 해?";

function errMsg(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

function IngestPanel({ onIngested }: { onIngested: () => void }) {
  const [docs, setDocs] = useState(SAMPLE_DOCS);
  const [result, setResult] = useState<IngestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const texts = docs.split("\n").map((s) => s.trim()).filter(Boolean);
      if (texts.length === 0) throw new Error("색인할 문서가 비었습니다");
      const r = await api<IngestResult>("/api/rag/ingest", { method: "POST", body: { texts } });
      setResult(r);
      onIngested();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h3" gutterBottom>문서 색인</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          한 줄에 하나의 문서를 입력하세요. 색인된 문서는 검색·증강에 사용됩니다.
        </Typography>
        <TextField
          multiline minRows={5} fullWidth value={docs}
          onChange={(e) => setDocs(e.target.value)}
          sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
        />
        <Box sx={{ mt: 2 }}>
          <Button variant="contained" onClick={run} disabled={busy}>
            {busy ? "색인 중…" : "색인하기"}
          </Button>
        </Box>

        {error && <ErrorView message={error} />}

        {result && (
          <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap", gap: 1 }}>
            <Chip size="small" color="success" variant="outlined" label={`색인 ${result.ingested}건`} />
            <Chip size="small" variant="outlined" label={`총 문서 ${result.total}건`} />
          </Stack>
        )}
      </CardContent>
    </Card>
  );
}

function RetrievePanel() {
  const [q, setQ] = useState(SAMPLE_QUERY);
  const [hits, setHits] = useState<RagHit[] | null>(null);
  const [augment, setAugment] = useState<AugmentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"query" | "augment" | null>(null);

  const runQuery = async () => {
    setBusy("query");
    setError(null);
    setAugment(null);
    try {
      const r = await api<QueryResult>("/api/rag/query", { method: "POST", body: { query: q } });
      setHits(r.hits);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(null);
    }
  };

  const runAugment = async () => {
    setBusy("augment");
    setError(null);
    try {
      const r = await api<AugmentResult>("/api/rag/augment", { method: "POST", body: { query: q } });
      setAugment(r);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h3" gutterBottom>검색 · 증강</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          질의로 top-k 문서를 검색하거나, 컨텍스트가 주입된 messages(서빙 전달 형태)를 미리 봅니다.
        </Typography>
        <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1 }}>
          <TextField
            fullWidth size="small" value={q} label="질의"
            onChange={(e) => setQ(e.target.value)}
            sx={{ flex: 1, minWidth: 240 }}
          />
          <Button variant="contained" onClick={runQuery} disabled={busy !== null}>
            {busy === "query" ? "검색 중…" : "검색"}
          </Button>
          <Button variant="outlined" onClick={runAugment} disabled={busy !== null}>
            {busy === "augment" ? "생성 중…" : "컨텍스트 주입 미리보기"}
          </Button>
        </Stack>

        {error && <ErrorView message={error} />}

        {hits && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" fontWeight={600} sx={{ mb: 1 }}>검색 결과</Typography>
            {hits.length === 0 ? (
              <EmptyView message="검색 결과가 없습니다. 먼저 문서를 색인하세요." />
            ) : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ width: 90 }} align="right">점수</TableCell>
                    <TableCell>문서</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {hits.map((h) => (
                    <TableRow key={h.id}>
                      <TableCell align="right" sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                        {h.score.toFixed(4)}
                      </TableCell>
                      <TableCell>{h.text}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </Box>
        )}

        {augment && (
          <Box sx={{ mt: 3 }}>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body2" fontWeight={600} sx={{ mb: 1 }}>
              서빙에 전달될 messages (system에 컨텍스트 주입됨)
            </Typography>
            <Stack spacing={1}>
              {augment.messages.map((m, i) => (
                <Card key={i} variant="outlined">
                  <CardContent sx={{ py: 1.5, "&:last-child": { pb: 1.5 } }}>
                    <Typography variant="caption" color="primary" fontWeight={700}>{m.role}</Typography>
                    <Typography variant="body2" sx={{ mt: 0.5, whiteSpace: "pre-wrap" }}>{m.content}</Typography>
                  </CardContent>
                </Card>
              ))}
            </Stack>
            {augment.used.length > 0 && (
              <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap", gap: 1 }}>
                <Typography variant="caption" color="text.secondary" sx={{ alignSelf: "center" }}>사용 문서:</Typography>
                {augment.used.map((u) => (
                  <Chip key={u.id} size="small" variant="outlined" label={`${u.id} · ${u.score.toFixed(4)}`} />
                ))}
              </Stack>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default function Rag() {
  const { data, loading, error, reload } = useApi<RagStats>("/api/rag/stats");

  return (
    <>
      <PageHeader
        title="RAG 지식베이스"
        subtitle="문서를 색인하고 질의로 검색·증강 — 서빙·평가가 같은 경로로 컨텍스트를 주입"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      {data && (
        <SummaryTiles stats={[
          { label: "색인 문서", value: data.documents, hint: data.collection, icon: MenuBookOutlined },
          { label: "임베딩 차원", value: data.dim, hint: data.embedder, icon: LayersOutlined },
          { label: "top_k", value: data.top_k, hint: data.backend, icon: SearchOutlined },
        ]} />
      )}

      {error && <ErrorView message={error} />}

      {loading && !data && <Loading />}

      <IngestPanel onIngested={() => reload()} />
      <RetrievePanel />
    </>
  );
}

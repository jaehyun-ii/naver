import { useState } from "react";
import {
  Box, Button, Card, CardContent, Chip, Divider, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
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

interface RagCollection {
  name: string;
  points: number;
  dim: number | null;
  active: boolean;
}

interface CollectionsInfo {
  serving: {
    backend: string; collection: string; embedder: string; embedding_model: string;
    query_prompt: string | null; reranker_model: string | null; top_k: number;
  };
  collections: RagCollection[];
  error?: string;
}

interface ChunkDoc {
  idx: number; title: string; family: string; n_chunks: number; n_articles: number;
}

interface ChunkSoc {
  soc: string; label: string; docs: ChunkDoc[];
}

const SAMPLE_DOCS = `선급증서 유효기간이 지나면 즉시 선급기관에 재검사를 신청해 갱신한다.
FAT 불합격 시 부적합 항목을 시정조치 요구서로 발행하고 재시험한다.
도크 진수 전 선급기관 입회 검사를 통과해야 한다.`;

const SAMPLE_QUERY = "선급증서 만료되면 어떻게 해?";

function errMsg(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

/** 벡터 DB 인덱싱 현황 — Qdrant 컬렉션 목록과 현재 서빙 검색 스택. */
function IndexingPanel() {
  const { data, loading, error } = useApi<CollectionsInfo>("/api/rag/collections");
  if (loading && !data) return <Loading />;
  if (error) return <ErrorView message={error} />;
  if (!data) return null;
  const s = data.serving;
  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h3" gutterBottom>벡터 DB 인덱싱</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          청킹된 조 단위 문서가 임베딩되어 Qdrant 컬렉션에 적재됩니다. 서빙 검색은 아래 활성 컬렉션을 사용합니다.
        </Typography>
        <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: "wrap", gap: 1 }}>
          <Chip size="small" color="primary" label={`임베더: ${s.embedding_model}`} />
          {s.query_prompt && <Chip size="small" variant="outlined" label={`질의 프롬프트: ${s.query_prompt}`} />}
          {s.reranker_model && <Chip size="small" color="secondary" label={`리랭커: ${s.reranker_model}`} />}
          <Chip size="small" variant="outlined" label={`백엔드: ${s.backend}`} />
          <Chip size="small" variant="outlined" label={`top_k: ${s.top_k}`} />
        </Stack>
        {data.error ? (
          <Typography variant="body2" color="text.secondary">
            Qdrant 미가용: {data.error}
          </Typography>
        ) : data.collections.length === 0 ? (
          <EmptyView message="Qdrant 컬렉션이 없습니다." />
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>컬렉션</TableCell>
                <TableCell align="right">포인트(벡터)</TableCell>
                <TableCell align="right">차원</TableCell>
                <TableCell>상태</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data.collections.map((c) => (
                <TableRow key={c.name} selected={c.active}>
                  <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{c.name}</TableCell>
                  <TableCell align="right">{c.points.toLocaleString()}</TableCell>
                  <TableCell align="right">{c.dim ?? "-"}</TableCell>
                  <TableCell>
                    {c.active && <Chip size="small" color="success" label="서빙 활성" />}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

/** 청킹 현황 — 선급별 문서·조·청크 수(리뷰 DB). 상세는 청킹 리뷰 페이지로. */
function ChunkingPanel() {
  const { data, loading, error } = useApi<ChunkSoc[]>("/api/review/docs");
  if (loading && !data) return <Loading />;
  if (error) {
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>청킹 현황</Typography>
          <Typography variant="body2" color="text.secondary">{error}</Typography>
        </CardContent>
      </Card>
    );
  }
  if (!data) return null;
  const totals = data.flatMap((s) => s.docs).reduce(
    (a, d) => ({ chunks: a.chunks + d.n_chunks, articles: a.articles + d.n_articles, docs: a.docs + 1 }),
    { chunks: 0, articles: 0, docs: 0 });
  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
          <Typography variant="h3">청킹 현황</Typography>
          <Button size="small" component={RouterLink} to="/review">청킹 리뷰로</Button>
        </Stack>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          원본 PDF → MinerU 추출 → 도메인 청커(조/항/호 구조 보존) → 인덱싱. 문서 {totals.docs}건 ·
          조 단위 {totals.articles.toLocaleString()}건 · 청크 {totals.chunks.toLocaleString()}건.
        </Typography>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>선급</TableCell>
              <TableCell>문서</TableCell>
              <TableCell align="right">조(article)</TableCell>
              <TableCell align="right">청크</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.map((s) => s.docs.map((d, i) => (
              <TableRow key={`${s.soc}-${d.idx}`}>
                <TableCell>{i === 0 ? `${s.label} (${s.soc})` : ""}</TableCell>
                <TableCell>{d.title}</TableCell>
                <TableCell align="right">{d.n_articles.toLocaleString()}</TableCell>
                <TableCell align="right">{d.n_chunks.toLocaleString()}</TableCell>
              </TableRow>
            )))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
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
        subtitle="문서 청킹 → 벡터 DB 인덱싱 → 검색·증강 — 서빙·평가가 같은 경로로 컨텍스트를 주입"
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

      <ChunkingPanel />
      <IndexingPanel />
      <IngestPanel onIngested={() => reload()} />
      <RetrievePanel />
    </>
  );
}

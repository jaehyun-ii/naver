import { useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  Alert, Box, Button, Card, CardContent, Chip, Dialog, DialogContent, DialogTitle,
  Divider, MenuItem, Stack, Tab, Tabs, Table, TableBody,
  TableCell, TableHead, TableRow, TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import DescriptionOutlined from "@mui/icons-material/DescriptionOutlined";
import CheckCircleOutline from "@mui/icons-material/CheckCircleOutline";
import LayersOutlined from "@mui/icons-material/LayersOutlined";
import ScatterPlotOutlined from "@mui/icons-material/ScatterPlot";
import PlayArrowOutlined from "@mui/icons-material/PlayArrow";
import UploadFileOutlined from "@mui/icons-material/UploadFileOutlined";
import FactCheckOutlined from "@mui/icons-material/FactCheckOutlined";
import { PageHeader } from "../components/PageHeader";
import { AuthImage } from "../components/AuthImage";
import { Loading, ErrorView } from "../components/StateViews";
import { SummaryTiles, RichEmpty } from "../components/SummaryTiles";
import { useApi } from "../hooks/useApi";
import { useAutoRefresh } from "../hooks/useAutoRefresh";
import { api, ApiError, apiUpload, apiBlob } from "../api";

interface Document {
  id: string;
  name: string;
  family: string;
  version: number;
  status: "pending" | "processing" | "processed" | "failed";
  content_list_key: string | null;
  chunks_key: string | null;
  vector_collection: string | null;
  n_chunks: number;
  n_vectors: number;
  error: string | null;
}

interface Block { type?: string; text?: string; page_idx?: number; text_level?: number; img_path?: string }
interface ContentListResp { total: number; types: Record<string, number>; blocks: Block[] }
interface Chunk {
  chunk_type?: string; section_path?: string[]; content?: string; article_no?: string;
  pages?: number[]; table_html?: string; table_caption?: string; caption?: string; image_path?: string;
}
interface ChunksResp { total: number; types: Record<string, number>; chunks: Chunk[] }

const STATUS_COLOR: Record<string, "default" | "info" | "success" | "error" | "warning"> = {
  pending: "warning", processing: "info", processed: "success", failed: "error",
};
const STATUS_LABEL: Record<string, string> = {
  pending: "대기", processing: "처리 중", processed: "완료", failed: "실패",
};
const FAMILIES = ["auto", "kr_rule", "nk_rule", "lr_code", "dnv_cg", "bv_rule", "iacs", "abs_guide"];

export default function Documents() {
  const { data, loading, error, reload } = useApi<Document[]>("/api/documents");
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [family, setFamily] = useState("auto");
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // 아티팩트 상세(원문·MinerU·청킹)
  const [detail, setDetail] = useState<Document | null>(null);
  const [tab, setTab] = useState(0);
  const [cl, setCl] = useState<ContentListResp | null>(null);
  const [ck, setCk] = useState<ChunksResp | null>(null);
  const [ckType, setCkType] = useState<string | null>(null);   // 청크 유형 필터(표·그림은 뒤쪽에 몰려 서버 필터 필요)
  const [detailBusy, setDetailBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);

  const fetchChunks = async (d: Document, type: string | null) => {
    setCkType(type); setDetailBusy(true);
    try {
      const q = type ? `&chunk_type=${encodeURIComponent(type)}` : "";
      setCk(await api<ChunksResp>(`/api/documents/${encodeURIComponent(d.id)}/chunks?limit=100${q}`));
    } catch { /* 프리뷰 실패는 무시 */ } finally { setDetailBusy(false); }
  };

  const openDetail = async (d: Document) => {
    setDetail(d); setTab(0); setCl(null); setCk(null); setCkType(null);
    setPdfUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null; });
    // 원문 PDF 프리페치(인라인 뷰어). 모든 상태에서 원본은 존재.
    apiBlob(`/api/documents/${encodeURIComponent(d.id)}/raw`)
      .then((blob) => setPdfUrl(URL.createObjectURL(blob)))
      .catch(() => setPdfUrl(null));
    if (d.status !== "processed") return;
    setDetailBusy(true);
    try {
      const [a, b] = await Promise.all([
        api<ContentListResp>(`/api/documents/${encodeURIComponent(d.id)}/content-list?limit=100`),
        api<ChunksResp>(`/api/documents/${encodeURIComponent(d.id)}/chunks?limit=100`),
      ]);
      setCl(a); setCk(b);
    } catch { /* 프리뷰 실패는 무시 */ } finally { setDetailBusy(false); }
  };

  const closeDetail = () => {
    setPdfUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null; });
    setDetail(null);
  };

  const docs = data ?? [];
  const anyProcessing = docs.some((d) => d.status === "processing");
  // 처리 중이면 3초 간격 자동 갱신
  useAutoRefresh(() => reload(), 3000, anyProcessing);

  const register = async () => {
    if (!path.trim()) return;
    setBusy(true); setFormError(null); setNotice(null);
    try {
      const d = await api<Document>("/api/documents", {
        method: "POST",
        body: { path: path.trim(), name: name.trim() || undefined, family },
      });
      setNotice(`${d.name} (v${d.version}) — ${d.status === "processed" ? "이미 처리 완료" : "등록 완료 · 처리 시작됨"}`);
      setPath(""); setName("");
      reload();
    } catch (e) {
      setFormError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    } finally { setBusy(false); }
  };

  const uploadFile = async (file: File) => {
    setUploadBusy(true); setFormError(null); setNotice(null);
    try {
      const form = new FormData();
      form.append("file", file);
      if (name.trim()) form.append("name", name.trim());
      form.append("family", family);
      const d = await apiUpload<Document>("/api/documents/upload", form);
      setNotice(`${d.name} (v${d.version}) — ${d.status === "processed" ? "이미 처리 완료" : "업로드 완료 · 처리 시작됨"}`);
      reload();
    } catch (e) {
      setFormError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    } finally { setUploadBusy(false); }
  };

  const process = async (id: string) => {
    setFormError(null);
    try {
      await api(`/api/documents/${encodeURIComponent(id)}/process`, { method: "POST" });
      reload();
    } catch (e) {
      setFormError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    }
  };

  return (
    <>
      <PageHeader
        title="문서 관리"
        subtitle="원본 PDF 등록 → ETL(MinerU) → 도메인 청킹 → Qdrant 벡터 적재. 인제스트 파이프라인 상태를 관리합니다."
        action={
          <Tooltip title="새로고침"><IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton></Tooltip>
        }
      />

      {docs.length > 0 && (
        <SummaryTiles stats={[
          { label: "문서", value: docs.length, hint: "원본 PDF", icon: DescriptionOutlined },
          { label: "처리 완료", value: docs.filter((d) => d.status === "processed").length, icon: CheckCircleOutline, accent: "success.main" },
          { label: "총 청크", value: docs.reduce((a, d) => a + d.n_chunks, 0), icon: LayersOutlined },
          { label: "총 벡터", value: docs.reduce((a, d) => a + d.n_vectors, 0), hint: "Qdrant 색인", icon: ScatterPlotOutlined, accent: "primary.main" },
        ]} />
      )}

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>문서 등록</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            PDF를 <strong>업로드</strong>하거나 서버 경로로 등록합니다(멱등 · sha256 중복 감지). 선급(family) 미지정 시 자동 판별.
            <strong> 등록 즉시 ETL→청킹→벡터화가 자동 실행</strong>됩니다.
          </Typography>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
            <TextField label="문서명 (선택)" size="small" value={name} onChange={(e) => setName(e.target.value)} sx={{ width: 180 }} />
            <TextField select label="선급(family)" size="small" value={family} onChange={(e) => setFamily(e.target.value)} sx={{ width: 150 }}>
              {FAMILIES.map((f) => <MenuItem key={f} value={f}>{f}</MenuItem>)}
            </TextField>
            <Button component="label" variant="contained" startIcon={<UploadFileOutlined />} disabled={uploadBusy}>
              {uploadBusy ? "업로드 중…" : "PDF 업로드"}
              <input hidden type="file" accept="application/pdf"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) uploadFile(f); e.target.value = ""; }} />
            </Button>
          </Stack>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} sx={{ mt: 2 }} alignItems={{ md: "center" }}>
            <TextField label="또는 서버 PDF 경로" size="small" value={path} onChange={(e) => setPath(e.target.value)}
              placeholder="/home/jaehyun/Dev/naver/data/KR_15.pdf" sx={{ flex: 1, minWidth: 260 }} />
            <Button variant="outlined" onClick={register} disabled={busy || !path.trim()}>{busy ? "등록 중…" : "경로 등록"}</Button>
          </Stack>
          {notice && <Alert severity="success" sx={{ mt: 2 }}>{notice}</Alert>}
          {formError && <Box sx={{ mt: 2 }}><ErrorView message={formError} /></Box>}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>문서 목록</Typography>
          {loading && !data ? (
            <Loading />
          ) : error ? (
            <ErrorView message={error} />
          ) : docs.length === 0 ? (
            <RichEmpty icon={DescriptionOutlined} title="등록된 문서가 없습니다"
              hint="위에서 원본 PDF 경로를 등록한 뒤 '처리 실행'하면 ETL→청킹→벡터DB까지 자동으로 진행됩니다." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>문서</TableCell>
                  <TableCell>선급</TableCell>
                  <TableCell>상태</TableCell>
                  <TableCell align="right">청크</TableCell>
                  <TableCell align="right">벡터</TableCell>
                  <TableCell>컬렉션</TableCell>
                  <TableCell align="right">액션</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {docs.map((d) => (
                  <TableRow key={d.id} hover>
                    <TableCell>
                      <Typography variant="body2" fontWeight={600}>{d.name}</Typography>
                      <Typography variant="caption" color="text.secondary">v{d.version}</Typography>
                    </TableCell>
                    <TableCell><Chip size="small" variant="outlined" label={d.family} /></TableCell>
                    <TableCell>
                      <Chip size="small" color={STATUS_COLOR[d.status] ?? "default"} label={STATUS_LABEL[d.status] ?? d.status} />
                      {d.status === "failed" && d.error && (
                        <Tooltip title={d.error}><Typography variant="caption" color="error.main" sx={{ ml: 1, cursor: "help" }}>사유</Typography></Tooltip>
                      )}
                    </TableCell>
                    <TableCell align="right">{d.n_chunks || "—"}</TableCell>
                    <TableCell align="right">{d.n_vectors || "—"}</TableCell>
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{d.vector_collection ?? "—"}</TableCell>
                    <TableCell align="right">
                      <Stack direction="row" spacing={1} justifyContent="flex-end">
                        <Button size="small" onClick={() => openDetail(d)}>상세</Button>
                        <Button size="small" variant="outlined" startIcon={<PlayArrowOutlined />}
                          disabled={d.status === "processing"} onClick={() => process(d.id)}>
                          {d.status === "processed" ? "재처리" : "처리 실행"}
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

      <Dialog open={!!detail} onClose={closeDetail} fullWidth maxWidth="md">
        <DialogTitle>
          {detail?.name}
          <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>v{detail?.version} · {detail?.family}</Typography>
        </DialogTitle>
        <DialogContent dividers>
          <Tabs value={tab} onChange={(_, v: number) => setTab(v)} sx={{ mb: 2 }}>
            <Tab label="① 원문 (PDF)" />
            <Tab label="② MinerU 결과" />
            <Tab label="③ 청킹 결과" />
          </Tabs>

          {detail && detail.status !== "processed" && (
            <Alert severity="info">아직 처리되지 않았습니다. 목록에서 "처리 실행"으로 ETL→청킹→벡터화를 돌리세요.</Alert>
          )}

          {tab === 0 && detail && (
            <Stack spacing={1}>
              {[["문서명", detail.name], ["버전", `v${detail.version}`], ["선급(family)", detail.family],
                ["상태", STATUS_LABEL[detail.status] ?? detail.status], ["원본 키", detail.id],
                ["청크/벡터", `${detail.n_chunks} / ${detail.n_vectors}`],
                ["벡터 컬렉션", detail.vector_collection ?? "—"]].map(([k, v]) => (
                <Stack key={k} direction="row" spacing={2}>
                  <Typography variant="body2" color="text.secondary" sx={{ width: 110, flexShrink: 0 }}>{k}</Typography>
                  <Typography variant="body2" sx={{ fontFamily: "ui-monospace, monospace", wordBreak: "break-all" }}>{v}</Typography>
                </Stack>
              ))}
              <Divider sx={{ my: 1 }} />
              {pdfUrl ? (
                <Box component="iframe" src={pdfUrl} title="원문 PDF"
                  sx={{ width: "100%", height: 520, border: 1, borderColor: "divider", borderRadius: 1 }} />
              ) : (
                <Typography variant="caption" color="text.secondary">원문 PDF 로딩 중… (또는 원본을 찾을 수 없음)</Typography>
              )}
              <Button component={RouterLink} to="/storage" variant="text" size="small" sx={{ alignSelf: "flex-start" }}>
                스토리지에서 열기 (documents-raw)
              </Button>
            </Stack>
          )}

          {tab === 1 && (detailBusy && !cl ? <Loading /> : cl ? (
            <Stack spacing={1.5}>
              <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                <Chip size="small" color="primary" label={`총 ${cl.total} 블록`} />
                {Object.entries(cl.types).map(([t, n]) => <Chip key={t} size="small" variant="outlined" label={`${t} ${n}`} />)}
              </Stack>
              <Box sx={{ maxHeight: 420, overflowY: "auto", border: 1, borderColor: "divider", borderRadius: 1 }}>
                {cl.blocks.map((b, i) => (
                  <Box key={i} sx={{ px: 1.5, py: 0.75, borderBottom: 1, borderColor: "divider" }}>
                    <Stack direction="row" spacing={1} alignItems="baseline">
                      <Chip size="small" variant="outlined" label={b.type ?? "?"} sx={{ height: 18, fontSize: 11 }} />
                      <Typography variant="caption" color="text.secondary">p.{(b.page_idx ?? 0) + 1}</Typography>
                    </Stack>
                    {b.text && <Typography variant="body2" sx={{ mt: 0.25, whiteSpace: "pre-wrap" }}>{b.text.slice(0, 300)}</Typography>}
                    {b.img_path && <Typography variant="caption" color="text.secondary">🖼 {b.img_path}</Typography>}
                  </Box>
                ))}
              </Box>
            </Stack>
          ) : <Alert severity="info">MinerU 결과물이 없습니다(미처리).</Alert>)}

          {tab === 2 && (detailBusy && !ck ? <Loading /> : ck ? (
            <Stack spacing={1.5}>
              <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap alignItems="center">
                <Chip size="small" color={ckType ? "default" : "success"} label={`총 ${ck.total} 청크`}
                  onClick={() => detail && fetchChunks(detail, null)} sx={{ cursor: "pointer" }} />
                {Object.entries(ck.types).map(([t, n]) => (
                  <Chip key={t} size="small" variant={ckType === t ? "filled" : "outlined"}
                    color={ckType === t ? "primary" : "default"} label={`${t} ${n}`}
                    onClick={() => detail && fetchChunks(detail, t)} sx={{ cursor: "pointer" }} />
                ))}
                <Box sx={{ flexGrow: 1 }} />
                <Button component={RouterLink} to="/review" size="small" startIcon={<FactCheckOutlined />}>
                  청킹 리뷰에서 대조·검색
                </Button>
              </Stack>
              <Box sx={{ maxHeight: 420, overflowY: "auto", border: 1, borderColor: "divider", borderRadius: 1 }}>
                {ck.chunks.map((c, i) => (
                  <Box key={i} sx={{ px: 1.5, py: 0.75, borderBottom: 1, borderColor: "divider" }}>
                    <Stack direction="row" spacing={1} alignItems="baseline">
                      <Chip size="small" variant="outlined" label={c.chunk_type ?? "?"} sx={{ height: 18, fontSize: 11 }} />
                      {(c.pages?.length ?? 0) > 0 && (
                        <Typography variant="caption" color="text.secondary">p.{(c.pages![0] ?? 0) + 1}</Typography>
                      )}
                      <Typography variant="caption" color="text.secondary" sx={{ fontFamily: "ui-monospace, monospace" }} noWrap>
                        {(c.section_path ?? []).join(" > ")}
                      </Typography>
                    </Stack>
                    {(c.table_caption || c.caption) && (
                      <Typography variant="caption" sx={{ display: "block", mt: 0.25, fontWeight: 600 }}>
                        {c.table_caption || c.caption}
                      </Typography>
                    )}
                    {c.chunk_type === "table" && c.table_html ? (
                      <Box sx={{ mt: 0.5, maxHeight: 300, overflow: "auto",
                                 "& table": { borderCollapse: "collapse", fontSize: 12 },
                                 "& td, & th": { border: "1px solid", borderColor: "divider", p: 0.5 } }}
                        dangerouslySetInnerHTML={{ __html: c.table_html }} />
                    ) : c.chunk_type === "figure" && c.image_path && detail ? (
                      <Box sx={{ mt: 0.5 }}>
                        <AuthImage
                          src={`/api/documents/${encodeURIComponent(detail.id)}/asset?path=${encodeURIComponent(c.image_path)}`}
                          alt={c.caption || c.image_path} />
                      </Box>
                    ) : (
                      c.content && <Typography variant="body2" sx={{ mt: 0.25, whiteSpace: "pre-wrap" }}>{c.content.slice(0, 300)}</Typography>
                    )}
                  </Box>
                ))}
              </Box>
            </Stack>
          ) : <Alert severity="info">청킹 결과가 없습니다(미처리).</Alert>)}
        </DialogContent>
      </Dialog>
    </>
  );
}

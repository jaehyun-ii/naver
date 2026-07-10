import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Accordion, AccordionDetails, AccordionSummary, Box, Chip, InputAdornment,
  MenuItem, Paper, Select, Stack, TextField, Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import SearchIcon from "@mui/icons-material/Search";
import FactCheckOutlined from "@mui/icons-material/FactCheckOutlined";
import DescriptionOutlined from "@mui/icons-material/DescriptionOutlined";
import LayersOutlined from "@mui/icons-material/LayersOutlined";
import { PageHeader } from "../components/PageHeader";
import { AuthImage } from "../components/AuthImage";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { SummaryTiles } from "../components/SummaryTiles";
import { useApi } from "../hooks/useApi";
import { api, getMasterKey } from "../api";

interface Doc { idx: number; title: string; family: string; n_chunks: number; n_articles: number; }
interface Society { soc: string; label: string; docs: Doc[]; }
interface Article {
  chunk_id: string; article_no: string; article_title: string;
  chapter_no: string; chapter_title: string; section_path: string[];
  content_tokens: number; n_units: number;
}
interface Unit {
  chunk_id: string; chunk_type: string; paragraph_no: string; item_no: string;
  sub_item_no: string;
  content: string; table_html: string; table_caption: string; caption: string;
  image_path: string;
}
// 백엔드 정렬 맵: unit_id → [blockStart, blockEnd, article_id]
type UnitMap = Record<string, [number, number, string]>;
interface Hit {
  score: number; section_path: string; matched: string;
  article_no: string; article_title: string; article_content: string;
}

// 원문 블록 분리 — 백엔드(review.py BLOCK_SPLIT)와 **동일 규칙**이어야 인덱스가 일치
const BLOCK_SPLIT =
  /\n{2,}|\n(?=\s*(?:\d{1,3}[.)]|\(\d{1,3}\)|[가-힣][.)]|\([가-힣]\)|[a-z][.)]|\([a-z]\)|[-●•▪])\s)/;

async function fetchText(path: string, signal?: AbortSignal): Promise<string> {
  const key = getMasterKey();
  const res = await fetch(path, { headers: key ? { "X-Master-Key": key } : {}, signal });
  return res.ok ? res.text() : "*(불러오기 실패)*";
}

interface LinkCtx {
  onUnitClick: (id: string) => void;
  hlChunk: string | null;
  openArt: string | null;
  regUnit: (id: string, el: HTMLElement | null) => void;
}

// 계층 트리 노드 — depth는 **스키마 필드**로 판정(청커가 항/호/목/세목을 분리·채움)
interface TNode { depth: number; text: string; chunkId: string; head: boolean; children: TNode[]; }

// 유닛 → depth: 항0 · 호1 · 목2 · 세목3 (sub_item_no="가"→목, "가.a"→세목)
function unitDepth(u: Unit): number {
  if (u.sub_item_no) return u.sub_item_no.includes(".") ? 3 : 2;
  return u.item_no ? 1 : 0;
}
function unitToNodes(u: Unit): Omit<TNode, "children">[] {
  return [{ depth: unitDepth(u), text: u.content, chunkId: u.chunk_id, head: true }];
}

function buildTree(flat: Omit<TNode, "children">[]): TNode[] {
  const root: TNode = { depth: -1, text: "", chunkId: "", head: false, children: [] };
  const stack: TNode[] = [root];
  for (const n of flat) {
    const node: TNode = { ...n, children: [] };
    while (stack.length > 1 && stack[stack.length - 1].depth >= node.depth) stack.pop();
    stack[stack.length - 1].children.push(node);
    stack.push(node);
  }
  return root.children;
}

function NodeRow({ node, prefix, link }: { node: TNode; prefix: string; link: LinkCtx }) {
  const on = link.hlChunk === node.chunkId;
  return (
    <Box
      ref={node.head ? (el: HTMLElement | null) => link.regUnit(node.chunkId, el) : undefined}
      onClick={() => link.onUnitClick(node.chunkId)}
      sx={{
        display: "flex", alignItems: "flex-start",
        bgcolor: on ? "primary.main" : "transparent",
        color: on ? "primary.contrastText" : "inherit",
        borderRadius: 1, py: 0.3, px: 0.5, cursor: "pointer",
        transition: "background-color .12s",
        "&:hover": { bgcolor: on ? "primary.main" : "action.hover" },
      }}
    >
      {prefix && (
        <Box component="span" sx={{ fontFamily: "monospace", whiteSpace: "pre", flexShrink: 0, color: on ? "inherit" : "text.disabled", fontSize: 13, lineHeight: 1.6 }}>
          {prefix}
        </Box>
      )}
      <Typography variant="body2" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.6, flex: 1 }}>
        {node.text}
      </Typography>
    </Box>
  );
}

// 호(depth1)부터 ├─/└─ 분기, 목/세목은 조상 가이드(│) 누적
function TreeNodes({ nodes, guides, link }: { nodes: TNode[]; guides: string[]; link: LinkCtx }) {
  return (
    <>
      {nodes.map((node, i) => {
        const isLast = i === nodes.length - 1;
        const prefix = guides.join("") + (isLast ? "└─ " : "├─ ");
        return (
          <Box key={i}>
            <NodeRow node={node} prefix={prefix} link={link} />
            {node.children.length > 0 && (
              <TreeNodes nodes={node.children} guides={[...guides, isLast ? "   " : "│  "]} link={link} />
            )}
          </Box>
        );
      })}
    </>
  );
}

function UnitTree({ units, link, soc, idx }: { units: Unit[]; link: LinkCtx; soc: string; idx: number }) {
  const text = units.filter((u) => u.chunk_type === "text");
  const tables = units.filter((u) => u.chunk_type === "table");
  const figures = units.filter((u) => u.chunk_type === "figure");
  // 항(depth0)은 루트로 연결선 없이, 그 하위(호/목/세목)를 트리로 렌더
  const tree = buildTree(text.flatMap(unitToNodes));
  return (
    <Stack spacing={0.15}>
      {tree.map((art, ai) => (
        <Box key={ai} sx={{ mt: ai > 0 ? 0.75 : 0 }}>
          <NodeRow node={art} prefix="" link={link} />
          <TreeNodes nodes={art.children} guides={[]} link={link} />
        </Box>
      ))}
      {tables.map((t) => (
        <Box key={t.chunk_id} sx={{ overflowX: "auto", my: 1 }}>
          {t.table_caption && (
            <Typography variant="caption" color="text.secondary">{t.table_caption}</Typography>
          )}
          <Box
            sx={{ "& table": { borderCollapse: "collapse", fontSize: 12 },
                   "& td,& th": { border: "1px solid", borderColor: "divider", p: 0.5 } }}
            dangerouslySetInnerHTML={{ __html: t.table_html || "" }}
          />
        </Box>
      ))}
      {figures.map((f) => (
        <Box key={f.chunk_id} sx={{ my: 1 }}>
          {f.caption && (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.25 }}>{f.caption}</Typography>
          )}
          {f.image_path ? (
            <AuthImage
              src={`/api/review/${encodeURIComponent(soc)}/${idx}/asset?path=${encodeURIComponent(f.image_path)}`}
              alt={f.caption || f.image_path} />
          ) : (
            <Typography variant="caption" color="text.secondary">🖼 (이미지 경로 없음)</Typography>
          )}
        </Box>
      ))}
    </Stack>
  );
}

function ArticleAccordion(
  { soc, idx, art, link }: { soc: string; idx: number; art: Article; link: LinkCtx },
) {
  const [units, setUnits] = useState<Unit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const path = art.section_path.slice(1).join(" › ");
  const load = useCallback(() => {
    if (units || loading) return;
    setLoading(true);
    api<Unit[]>(`/api/review/${soc}/${idx}/article/${encodeURIComponent(art.chunk_id)}`)
      .then(setUnits).finally(() => setLoading(false));
  }, [units, loading, soc, idx, art.chunk_id]);

  // md→청크: 이 조가 지목되면 자동으로 펼치고 로드
  useEffect(() => {
    if (link.openArt === art.chunk_id) { setOpen(true); load(); }
  }, [link.openArt, art.chunk_id, load]);

  const hit = link.hlChunk && units?.some((u) => u.chunk_id === link.hlChunk);
  return (
    <Accordion
      disableGutters expanded={open}
      onChange={(_, o) => { setOpen(o); if (o) load(); }}
      sx={{ "&:before": { display: "none" },
            borderLeft: "3px solid", borderColor: hit ? "primary.main" : "transparent" }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Box sx={{ minWidth: 0 }}>
          <Stack direction="row" spacing={1} alignItems="baseline">
            {art.article_no && (
              <Typography variant="subtitle2" color="primary" sx={{ fontWeight: 700 }}>{art.article_no}</Typography>
            )}
            <Typography variant="subtitle2" noWrap sx={{ fontWeight: 700 }}>
              {art.article_title || "(도입문)"}
            </Typography>
            <Chip label={`${art.n_units}단위`} size="small" variant="outlined" />
          </Stack>
          {path && (
            <Typography variant="caption" color="text.secondary" noWrap sx={{ display: "block" }}>{path}</Typography>
          )}
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        {loading && <Loading label="본문 불러오는 중…" />}
        {units && (units.length ? <UnitTree units={units} link={link} soc={soc} idx={idx} /> : <EmptyView message="내용 없음" />)}
      </AccordionDetails>
    </Accordion>
  );
}

export default function Review() {
  const { data: socs, loading, error } = useApi<Society[]>("/api/review/docs");
  const [soc, setSoc] = useState("");
  const [idx, setIdx] = useState<number>(0);
  const [md, setMd] = useState("");
  const [umap, setUmap] = useState<UnitMap>({});
  const [arts, setArts] = useState<Article[] | null>(null);
  const [busy, setBusy] = useState(false);

  const [hlBlocks, setHlBlocks] = useState<Set<number>>(new Set());
  const [hlChunk, setHlChunk] = useState<string | null>(null);
  const [openArt, setOpenArt] = useState<string | null>(null);
  // 의미 검색(bge-m3 → Qdrant → 조 확장)
  const [sq, setSq] = useState("");
  const [results, setResults] = useState<Hit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const doSearch = useCallback(() => {
    const q = sq.trim();
    if (q.length < 2) { setResults(null); return; }
    setSearching(true);
    api<{ hits: Hit[] }>(`/api/review/search?q=${encodeURIComponent(q)}&k=8`)
      .then((r) => setResults(r.hits)).catch(() => setResults([])).finally(() => setSearching(false));
  }, [sq]);
  const mdEls = useRef<Map<number, HTMLElement>>(new Map());
  const unitEls = useRef<Map<string, HTMLElement>>(new Map());

  useEffect(() => { if (socs && socs.length && !soc) setSoc(socs[0].soc); }, [socs, soc]);
  const society = socs?.find((s) => s.soc === soc);
  useEffect(() => { setIdx(0); }, [soc]);

  const mdBlocks = useMemo(
    () => md.split(BLOCK_SPLIT).map((t) => t.trim()).filter(Boolean),
    [md],
  );
  // 블록 인덱스 → 소속 유닛 id (md 클릭 시 역방향)
  const blockOwner = useMemo(() => {
    const owner: Record<number, string> = {};
    for (const [uid, v] of Object.entries(umap)) {
      for (let b = v[0]; b <= v[1]; b++) if (owner[b] === undefined) owner[b] = uid;
    }
    return owner;
  }, [umap]);

  const loadDoc = useCallback((s: string, i: number, signal?: AbortSignal) => {
    setBusy(true); setArts(null); setMd(""); setUmap({});
    setHlBlocks(new Set()); setHlChunk(null); setOpenArt(null);
    mdEls.current.clear(); unitEls.current.clear();
    Promise.all([
      fetchText(`/api/review/${s}/${i}/md`, signal).then(setMd),
      api<Article[]>(`/api/review/${s}/${i}/articles`, { signal }).then(setArts).catch(() => setArts([])),
      api<{ units: UnitMap }>(`/api/review/${s}/${i}/map`, { signal }).then((r) => setUmap(r.units)).catch(() => setUmap({})),
    ]).finally(() => setBusy(false));
  }, []);

  useEffect(() => {
    if (!soc || !society) return;
    const ctrl = new AbortController();
    loadDoc(soc, idx, ctrl.signal);
    return () => ctrl.abort();
  }, [soc, idx, society, loadDoc]);

  const setBlockRange = useCallback((v?: [number, number, string]) => {
    const s = new Set<number>();
    if (v) for (let b = v[0]; b <= v[1]; b++) s.add(b);
    setHlBlocks(s);
    return v ? v[0] : -1;
  }, []);

  // 청크 유닛 클릭 → 정렬맵으로 원문 블록 구간 하이라이트+스크롤
  const onUnitClick = useCallback((id: string) => {
    setHlChunk(id);
    const first = setBlockRange(umap[id]);
    if (first >= 0) mdEls.current.get(first)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [umap, setBlockRange]);

  const regUnit = useCallback((id: string, el: HTMLElement | null) => {
    if (el) unitEls.current.set(id, el); else unitEls.current.delete(id);
  }, []);

  // 원문 블록 클릭 → 소속 유닛 찾아 조 펼치고 하이라이트(접힌 조도)
  const onMdClick = useCallback((bi: number) => {
    const uid = blockOwner[bi];
    if (!uid) { setHlBlocks(new Set([bi])); setHlChunk(null); setOpenArt(null); return; }
    const v = umap[uid];
    setHlChunk(uid);
    setOpenArt(v[2]);
    setBlockRange(v);
    // 유닛이 로드되면 스크롤(로드 지연 대비 재시도)
    let tries = 0;
    const tick = () => {
      const el = unitEls.current.get(uid);
      if (el) el.scrollIntoView({ block: "center", behavior: "smooth" });
      else if (tries++ < 20) setTimeout(tick, 80);
    };
    tick();
  }, [blockOwner, umap, setBlockRange]);

  const link: LinkCtx = { onUnitClick, hlChunk, openArt, regUnit };

  if (loading) return <Loading />;
  if (error) return <ErrorView message={error} />;
  if (!socs) return <EmptyView message="데이터 없음 (scripts/build_review_db.py 실행 필요)" />;

  return (
    <Box>
      <PageHeader title="청킹 리뷰" subtitle="원문(.md) ↔ 청킹 대조 · 의미 검색(bge-m3) + 조 문맥 확장" />

      {(socs?.length ?? 0) > 0 && (
        <SummaryTiles stats={[
          { label: "선급", value: socs!.length, hint: "발행기관", icon: FactCheckOutlined },
          { label: "문서", value: socs!.reduce((a, s) => a + s.docs.length, 0), hint: "색인 대상", icon: DescriptionOutlined },
          { label: "총 청크", value: socs!.flatMap((s) => s.docs).reduce((a, d) => a + d.n_chunks, 0), hint: "parent+child", icon: LayersOutlined },
        ]} />
      )}
      <TextField
        size="small" fullWidth value={sq} placeholder="의미 검색 — 예: 단강품 충격시험 재시험 요건 (Enter)"
        onChange={(e) => setSq(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") doSearch(); if (e.key === "Escape") { setSq(""); setResults(null); } }}
        sx={{ mb: 1.5 }}
        InputProps={{ startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" /></InputAdornment> }}
      />
      {results !== null && (
        <Paper variant="outlined" sx={{ p: 1.5, mb: 1.5, maxHeight: "calc(100vh - 240px)", overflow: "auto" }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="subtitle2">검색 결과 {results.length}건 (매칭 조항 → 소속 조 전체)</Typography>
            <Chip label="닫기" size="small" onClick={() => { setResults(null); setSq(""); }} />
          </Stack>
          {searching && <Loading label="검색 중…" />}
          {results.map((h, i) => (
            <Accordion key={i} disableGutters sx={{ "&:before": { display: "none" }, mb: 0.5 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ minWidth: 0 }}>
                  <Stack direction="row" spacing={1} alignItems="baseline">
                    <Chip label={h.score.toFixed(3)} size="small" color="primary" variant="outlined" />
                    <Typography variant="subtitle2" sx={{ fontWeight: 700 }} noWrap>
                      {h.article_no} {h.article_title}
                    </Typography>
                  </Stack>
                  <Typography variant="caption" color="text.secondary" noWrap sx={{ display: "block" }}>
                    {h.section_path}
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 0.5, color: "primary.main" }} noWrap>
                    ▶ {h.matched}
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="caption" color="text.secondary">소속 조 전체 (문맥 확장)</Typography>
                <Typography variant="body2" sx={{ whiteSpace: "pre-wrap", mt: 0.5, lineHeight: 1.6 }}>
                  {h.article_content}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))}
          {!searching && !results.length && <EmptyView message="결과 없음" />}
        </Paper>
      )}
      <Stack direction="row" spacing={1.5} sx={{ mb: 1.5 }}>
        <Select size="small" value={soc} onChange={(e) => setSoc(e.target.value)} sx={{ minWidth: 200 }}>
          {socs.map((s) => <MenuItem key={s.soc} value={s.soc}>{s.label} ({s.docs.length})</MenuItem>)}
        </Select>
        <Select size="small" value={society ? idx : ""} onChange={(e) => setIdx(Number(e.target.value))}
          sx={{ minWidth: 360, flex: 1 }}>
          {society?.docs.map((d) => (
            <MenuItem key={d.idx} value={d.idx}>{d.title} · {d.n_articles}조</MenuItem>
          ))}
        </Select>
      </Stack>

      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1.5, height: "calc(100vh - 210px)" }}>
        <Paper variant="outlined" sx={{ overflow: "auto", p: 2 }}>
          <Typography variant="overline" color="text.secondary">원문 (.md) — 클릭 시 해당 청크로</Typography>
          <Stack spacing={0.25} sx={{ mt: 1 }}>
            {mdBlocks.map((t, i) => {
              const on = hlBlocks.has(i);
              return (
                <Box
                  key={i}
                  ref={(el: HTMLElement | null) => { if (el) mdEls.current.set(i, el); }}
                  onClick={() => onMdClick(i)}
                  sx={{
                    px: 1, py: 0.5, borderRadius: 1, cursor: "pointer",
                    bgcolor: on ? "warning.main" : "transparent",
                    color: on ? "warning.contrastText" : "inherit",
                    "&:hover": { bgcolor: on ? "warning.main" : "action.hover" },
                  }}
                >
                  <Typography variant="body2" sx={{ whiteSpace: "pre-wrap", fontFamily: "inherit", lineHeight: 1.55 }}>
                    {t}
                  </Typography>
                </Box>
              );
            })}
            {!mdBlocks.length && !busy && <Typography variant="body2" color="text.secondary">—</Typography>}
          </Stack>
        </Paper>

        <Paper variant="outlined" sx={{ overflow: "auto", p: 1 }}>
          <Typography variant="overline" color="text.secondary" sx={{ px: 1 }}>
            청킹 결과 (조 &gt; 항 &gt; 호) — 클릭 시 원문으로
          </Typography>
          {busy && !arts && <Loading label="조 목록 불러오는 중…" />}
          {arts && (arts.length
            ? arts.map((a) => <ArticleAccordion key={a.chunk_id} soc={soc} idx={idx} art={a} link={link} />)
            : <EmptyView message="조 없음" />)}
        </Paper>
      </Box>
    </Box>
  );
}

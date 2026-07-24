/** AIReg 생성 데이터셋 섹션 — 데이터셋 페이지(생성·관리)와 AIReg 페이지가 공유. */
import { useState } from "react";
import {
  Card, CardContent, Chip, Collapse, IconButton, MenuItem, Stack, Table,
  TableBody, TableCell, TableHead, TableRow, TextField, Typography,
} from "@mui/material";
import KeyboardArrowDown from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUp from "@mui/icons-material/KeyboardArrowUp";
import { Loading, ErrorView, EmptyView } from "./StateViews";
import { useApi } from "../hooks/useApi";
import { useAutoRefresh } from "../hooks/useAutoRefresh";

export const AIREG_STATUS_COLOR: Record<string, "success" | "warning" | "error" | "default"> = {
  ACCEPT: "success", REVIEW: "warning", REJECT: "error",
};

export const AIREG_TRACKS = [
  "spec", "applicability", "crossref", "def_link", "precedence",
  "unit_convert", "table_lookup", "hierarchy",
];

interface Overview {
  suite: string;
  tracks: Record<string, Record<string, number>>;
  split: { train: number; val: number } | null;
  batch_tail: string[];
  gen_progress: { tuned: number; base: number };
  judge_summary: Record<string, { n: number; avg: number; ge4_ratio: number }>;
  summary: Record<string, unknown> | null;
}

interface QaEvidence { section_path: string; quote: string }

interface QaRow {
  question_id: string; track: string; status: string;
  review_reasons: string[]; question: string; gold_answer: string;
  evidence: QaEvidence[]; ts: string;
}

export function AiregOverviewSection({ suite, live }: { suite: string; live: boolean }) {
  const { data, loading, error, reload } = useApi<Overview>(`/api/aireg/overview?suite=${suite}`, [suite]);
  useAutoRefresh(reload, 5000, live);
  if (loading && !data) return <Loading />;
  if (error) return <ErrorView message={error} />;
  if (!data) return <EmptyView message="데이터 없음" />;
  const totals = Object.values(data.tracks).reduce(
    (a, t) => ({
      total: a.total + (t.total ?? 0), ACCEPT: a.ACCEPT + (t.ACCEPT ?? 0),
      REVIEW: a.REVIEW + (t.REVIEW ?? 0), REJECT: a.REJECT + (t.REJECT ?? 0),
    }), { total: 0, ACCEPT: 0, REVIEW: 0, REJECT: 0 });
  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
        {[["생성 QA", `${totals.total}건`], ["ACCEPT", `${totals.ACCEPT}`],
          ["REVIEW", `${totals.REVIEW}`], ["REJECT", `${totals.REJECT}`],
          ["SFT 분할", data.split ? `${data.split.train}/${data.split.val}` : "-"],
          ["생성평가(튜닝/베이스)", `${data.gen_progress.tuned}/${data.gen_progress.base}`],
        ].map(([k, v]) => (
          <Card key={k} sx={{ minWidth: 150 }}>
            <CardContent sx={{ py: 1.5 }}>
              <Typography variant="caption" color="text.secondary">{k}</Typography>
              <Typography variant="h6">{v}</Typography>
            </CardContent>
          </Card>
        ))}
        {Object.entries(data.judge_summary).map(([tag, j]) => (
          <Card key={tag} sx={{ minWidth: 170 }}>
            <CardContent sx={{ py: 1.5 }}>
              <Typography variant="caption" color="text.secondary">
                judge {tag === "tuned" ? "튜닝" : "베이스"} (n={j.n})
              </Typography>
              <Typography variant="h6">{j.avg} / 5</Typography>
              <Typography variant="caption">4점 이상 {(j.ge4_ratio * 100).toFixed(0)}%</Typography>
            </CardContent>
          </Card>
        ))}
      </Stack>
      <Card><CardContent>
        <Typography variant="subtitle2" gutterBottom>트랙별 현황</Typography>
        <Table size="small">
          <TableHead><TableRow>
            <TableCell>트랙</TableCell><TableCell>전체</TableCell>
            <TableCell>ACCEPT</TableCell><TableCell>REVIEW</TableCell><TableCell>REJECT</TableCell>
          </TableRow></TableHead>
          <TableBody>
            {Object.entries(data.tracks).map(([tr, t]) => (
              <TableRow key={tr}>
                <TableCell>{tr}</TableCell><TableCell>{t.total}</TableCell>
                <TableCell>{t.ACCEPT ?? 0}</TableCell><TableCell>{t.REVIEW ?? 0}</TableCell>
                <TableCell>{t.REJECT ?? 0}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent></Card>
      {data.summary && (
        <Card><CardContent>
          <Typography variant="subtitle2" gutterBottom>파이프라인 요약</Typography>
          <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontSize: 12 }}>
            {JSON.stringify(data.summary, null, 2)}
          </pre>
        </CardContent></Card>
      )}
    </Stack>
  );
}

/** QA 행 — 요약(질문·정답) + 확장 상세(전체 답변·근거 발췌·거절/리뷰 사유 전문). */
function QaRowView({ r }: { r: QaRow }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <TableRow hover onClick={() => setOpen(!open)} sx={{ cursor: "pointer" }}>
        <TableCell>
          <IconButton size="small">{open ? <KeyboardArrowUp /> : <KeyboardArrowDown />}</IconButton>
        </TableCell>
        <TableCell>
          <Chip size="small" color={AIREG_STATUS_COLOR[r.status] ?? "default"} label={r.status} />
          {r.review_reasons.slice(0, 1).map((x) => (
            <Typography key={x} variant="caption" display="block" color="text.secondary">
              {x.split(":")[0]}
            </Typography>))}
        </TableCell>
        <TableCell>{r.track}<Typography variant="caption" display="block" color="text.secondary">{r.ts.slice(11, 19)}</Typography></TableCell>
        <TableCell>
          <Typography variant="body2" noWrap sx={{ maxWidth: 560 }}>{r.question}</Typography>
          <Typography variant="body2" color="success.main" noWrap sx={{ mt: 0.5, maxWidth: 560 }}>
            {r.gold_answer}
          </Typography>
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell colSpan={4} sx={{ py: 0, border: 0 }}>
          <Collapse in={open} unmountOnExit>
            <Stack spacing={1} sx={{ py: 1.5, pl: 2, pr: 2 }}>
              <Typography variant="caption" color="text.secondary">질문 전문</Typography>
              <Typography variant="body2" whiteSpace="pre-wrap">{r.question}</Typography>
              <Typography variant="caption" color="success.main">정답(실제 답변) 전문</Typography>
              <Typography variant="body2" whiteSpace="pre-wrap">{r.gold_answer || "(없음)"}</Typography>
              {r.evidence.length > 0 && (
                <>
                  <Typography variant="caption" color="primary.main">근거 발췌 (evidence)</Typography>
                  {r.evidence.map((e, i) => (
                    <Stack key={i} spacing={0.25} sx={{ borderLeft: "3px solid", borderColor: "divider", pl: 1 }}>
                      <Typography variant="caption" color="text.secondary">{e.section_path}</Typography>
                      <Typography variant="body2" whiteSpace="pre-wrap">{e.quote}</Typography>
                    </Stack>
                  ))}
                </>
              )}
              {r.review_reasons.length > 0 && (
                <>
                  <Typography variant="caption" color="warning.main">
                    {r.status === "REJECT" ? "거절 사유" : "리뷰 사유"} 전문
                  </Typography>
                  {r.review_reasons.map((x, i) => (
                    <Typography key={i} variant="body2" whiteSpace="pre-wrap">· {x}</Typography>
                  ))}
                </>
              )}
            </Stack>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

export function AiregQaSection({ suite, live }: { suite: string; live: boolean }) {
  const [status, setStatus] = useState("");
  const [track, setTrack] = useState("");
  const qs = `suite=${suite}&limit=50${status ? `&status=${status}` : ""}${track ? `&track=${track}` : ""}`;
  const { data, loading, error, reload } = useApi<{ total: number; rows: QaRow[] }>(`/api/aireg/qa?${qs}`, [qs]);
  useAutoRefresh(reload, 7000, live);
  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2}>
        <TextField select size="small" label="상태" value={status} sx={{ width: 140 }}
          onChange={(e) => setStatus(e.target.value)}>
          <MenuItem value="">전체</MenuItem>
          {["ACCEPT", "REVIEW", "REJECT"].map((s) => <MenuItem key={s} value={s}>{s}</MenuItem>)}
        </TextField>
        <TextField select size="small" label="트랙" value={track} sx={{ width: 180 }}
          onChange={(e) => setTrack(e.target.value)}>
          <MenuItem value="">전체</MenuItem>
          {AIREG_TRACKS.map((t) => <MenuItem key={t} value={t}>{t}</MenuItem>)}
        </TextField>
        {data && <Chip label={`${data.total}건`} sx={{ alignSelf: "center" }} />}
        <Typography variant="caption" color="text.secondary" sx={{ alignSelf: "center" }}>
          원본(검수 형식) 표시 — [발췌]/[상황] 라벨은 학습 투입 시 자동 제거됩니다
        </Typography>
      </Stack>
      {loading && !data ? <Loading /> : error ? <ErrorView message={error} /> : (
        <Card><CardContent sx={{ p: 0 }}>
          <Table size="small">
            <TableHead><TableRow>
              <TableCell width={40} />
              <TableCell width={90}>상태</TableCell><TableCell width={110}>트랙</TableCell>
              <TableCell>질문 / 정답 (생성 시각순 · 행 클릭 시 근거·사유 상세)</TableCell>
            </TableRow></TableHead>
            <TableBody>
              {(data?.rows ?? []).map((r) => <QaRowView key={r.question_id + r.track} r={r} />)}
            </TableBody>
          </Table>
        </CardContent></Card>
      )}
    </Stack>
  );
}

/** 스위트 선택 훅 — 데이터셋·AIReg 페이지가 동일한 셀렉터 UI를 공유. */
export function useAiregSuite() {
  const [suite, setSuite] = useState("suite_v5");
  const { data } = useApi<string[]>("/api/aireg/suites");
  return { suite, setSuite, suites: data ?? ["suite_v5"] };
}

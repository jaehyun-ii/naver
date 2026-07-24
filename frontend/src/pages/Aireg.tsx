import { useState } from "react";
import {
  Box, Card, CardContent, Chip, Collapse, FormControlLabel, IconButton,
  MenuItem, Stack, Switch, Tab, Table, TableBody, TableCell, TableHead,
  TableRow, Tabs, TextField, Typography,
} from "@mui/material";
import KeyboardArrowDown from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUp from "@mui/icons-material/KeyboardArrowUp";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { useAutoRefresh } from "../hooks/useAutoRefresh";

const STATUS_COLOR: Record<string, "success" | "warning" | "error" | "default"> = {
  ACCEPT: "success", REVIEW: "warning", REJECT: "error",
};

interface Overview {
  suite: string;
  tracks: Record<string, Record<string, number>>;
  split: { train: number; val: number } | null;
  batch_tail: string[];
  gen_progress: { tuned: number; base: number };
  judge_summary: Record<string, { n: number; avg: number; ge4_ratio: number }>;
  summary: Record<string, unknown> | null;
}

interface QaRow {
  question_id: string; track: string; status: string;
  review_reasons: string[]; question: string; gold_answer: string; ts: string;
}

interface GenRow {
  source: string; track: string; question: string; gold: string;
  tuned_gen: string; base_gen: string;
  tuned_score: number | null; tuned_reason: string;
  base_score: number | null; base_reason: string;
  gold_label: string | null; tuned_label: string | null;
  label_match: boolean | null;
}

function ScoreChip({ v }: { v: number | null }) {
  if (v == null) return <Chip size="small" label="채점중" />;
  const color = v >= 4 ? "success" : v >= 3 ? "warning" : "error";
  return <Chip size="small" color={color} label={v.toFixed(0)} />;
}

function OverviewTab({ suite, live }: { suite: string; live: boolean }) {
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

function QaTab({ suite, live }: { suite: string; live: boolean }) {
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
          {["spec", "applicability", "crossref", "def_link", "precedence",
            "unit_convert", "table_lookup", "hierarchy"].map((t) => (
            <MenuItem key={t} value={t}>{t}</MenuItem>))}
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
              <TableCell width={90}>상태</TableCell><TableCell width={110}>트랙</TableCell>
              <TableCell>질문 / 정답 (생성 시각순)</TableCell>
            </TableRow></TableHead>
            <TableBody>
              {(data?.rows ?? []).map((r) => (
                <TableRow key={r.question_id + r.track}>
                  <TableCell>
                    <Chip size="small" color={STATUS_COLOR[r.status] ?? "default"} label={r.status} />
                    {r.review_reasons.slice(0, 1).map((x) => (
                      <Typography key={x} variant="caption" display="block" color="text.secondary">
                        {x.split(":")[0]}
                      </Typography>))}
                  </TableCell>
                  <TableCell>{r.track}<Typography variant="caption" display="block" color="text.secondary">{r.ts.slice(11, 19)}</Typography></TableCell>
                  <TableCell>
                    <Typography variant="body2">{r.question}</Typography>
                    <Typography variant="body2" color="success.main" sx={{ mt: 0.5 }}>{r.gold_answer}</Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent></Card>
      )}
    </Stack>
  );
}

function GenRowView({ r }: { r: GenRow }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <TableRow hover onClick={() => setOpen(!open)} sx={{ cursor: "pointer" }}>
        <TableCell width={40}>
          <IconButton size="small">{open ? <KeyboardArrowUp /> : <KeyboardArrowDown />}</IconButton>
        </TableCell>
        <TableCell width={110}>{r.track}</TableCell>
        <TableCell><Typography variant="body2" noWrap sx={{ maxWidth: 480 }}>{r.question}</Typography></TableCell>
        <TableCell width={70}><ScoreChip v={r.tuned_score} /></TableCell>
        <TableCell width={70}><ScoreChip v={r.base_score} /></TableCell>
        <TableCell width={110}>
          {r.gold_label && (
            <Chip size="small" color={r.label_match ? "success" : "error"}
              label={`${r.gold_label}→${r.tuned_label ?? "?"}`} />)}
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell colSpan={6} sx={{ py: 0, border: 0 }}>
          <Collapse in={open} unmountOnExit>
            <Stack spacing={1} sx={{ py: 1.5, pl: 2 }}>
              <Typography variant="caption" color="text.secondary">질문</Typography>
              <Typography variant="body2" whiteSpace="pre-wrap">{r.question}</Typography>
              <Typography variant="caption" color="success.main">정답</Typography>
              <Typography variant="body2" whiteSpace="pre-wrap">{r.gold}</Typography>
              <Typography variant="caption" color="primary.main">
                튜닝 응답 {r.tuned_score != null && `(judge ${r.tuned_score}점 — ${r.tuned_reason})`}
              </Typography>
              <Typography variant="body2" whiteSpace="pre-wrap">{r.tuned_gen || "(생성 대기)"}</Typography>
              <Typography variant="caption" color="text.secondary">
                베이스 응답 {r.base_score != null && `(judge ${r.base_score}점 — ${r.base_reason})`}
              </Typography>
              <Typography variant="body2" whiteSpace="pre-wrap" color="text.secondary">
                {r.base_gen || "(생성 대기)"}
              </Typography>
            </Stack>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

function GenEvalTab({ suite, live }: { suite: string; live: boolean }) {
  const [only, setOnly] = useState("all");
  const { data, loading, error, reload } = useApi<{ total: number; rows: GenRow[] }>(
    `/api/aireg/geneval?suite=${suite}&limit=40&only=${only}`, [suite, only]);
  useAutoRefresh(reload, 8000, live);
  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2}>
        <TextField select size="small" label="필터" value={only} sx={{ width: 200 }}
          onChange={(e) => setOnly(e.target.value)}>
          <MenuItem value="all">전체</MenuItem>
          <MenuItem value="scored">채점 완료만</MenuItem>
          <MenuItem value="mismatch">판정 불일치만</MenuItem>
        </TextField>
        {data && <Chip label={`${data.total}건`} sx={{ alignSelf: "center" }} />}
      </Stack>
      {loading && !data ? <Loading /> : error ? <ErrorView message={error} /> : (
        <Card><CardContent sx={{ p: 0 }}>
          <Table size="small">
            <TableHead><TableRow>
              <TableCell /><TableCell>트랙</TableCell><TableCell>질문</TableCell>
              <TableCell>튜닝</TableCell><TableCell>베이스</TableCell><TableCell>판정</TableCell>
            </TableRow></TableHead>
            <TableBody>
              {(data?.rows ?? []).map((r) => <GenRowView key={r.source} r={r} />)}
            </TableBody>
          </Table>
        </CardContent></Card>
      )}
    </Stack>
  );
}

function LogsTab({ suite, live }: { suite: string; live: boolean }) {
  const { data, loading, error, reload } = useApi<{ lines: string[] }>(
    `/api/aireg/logs?suite=${suite}&name=batch&lines=120`, [suite]);
  useAutoRefresh(reload, 5000, live);
  if (loading && !data) return <Loading />;
  if (error) return <ErrorView message={error} />;
  return (
    <Card><CardContent>
      <Typography variant="subtitle2" gutterBottom>배치 로그 (batch.log)</Typography>
      <Box component="pre" sx={{
        m: 0, p: 1.5, bgcolor: "background.default", borderRadius: 1,
        fontSize: 12, maxHeight: 520, overflow: "auto", whiteSpace: "pre-wrap",
      }}>
        {(data?.lines ?? []).join("\n") || "(로그 없음)"}
      </Box>
    </CardContent></Card>
  );
}

interface TrainExp { name: string; config: string; val_loss: number | null; n: number | null }

function TrainTab({ suite }: { suite: string }) {
  const { data, loading, error } = useApi<{ experiments: TrainExp[]; logs: string }>(
    `/api/aireg/train?suite=${suite}`, [suite]);
  if (loading && !data) return <Loading />;
  if (error) return <ErrorView message={error} />;
  const best = data?.experiments.find((e) => e.name !== "base32b");
  return (
    <Stack spacing={2}>
      <Card><CardContent>
        <Typography variant="subtitle2" gutterBottom>
          학습 실험 (val_loss 오름차순 · 32B LoRA · val 200)
          {best && <Chip size="small" color="success" sx={{ ml: 1 }} label={`최고: ${best.name} ${best.val_loss}`} />}
        </Typography>
        <Table size="small">
          <TableHead><TableRow>
            <TableCell>실험</TableCell><TableCell>설정</TableCell>
            <TableCell align="right">val_loss</TableCell>
          </TableRow></TableHead>
          <TableBody>
            {(data?.experiments ?? []).map((e) => (
              <TableRow key={e.name} selected={e.name === "main800"}>
                <TableCell>{e.name}</TableCell><TableCell>{e.config}</TableCell>
                <TableCell align="right">{e.val_loss ?? "-"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent></Card>
      <Card><CardContent>
        <Typography variant="subtitle2" gutterBottom>학습 실행 로그 (H100 동기화)</Typography>
        <Box component="pre" sx={{ m: 0, p: 1.5, bgcolor: "background.default", borderRadius: 1,
          fontSize: 12, maxHeight: 420, overflow: "auto", whiteSpace: "pre-wrap" }}>
          {data?.logs || "(로그 없음)"}
        </Box>
      </CardContent></Card>
    </Stack>
  );
}

function pct(v?: number | null) { return v == null ? "-" : `${(v * 100).toFixed(1)}%`; }

function RetrievalTab({ suite }: { suite: string }) {
  const { data, loading, error } = useApi<Record<string, any>>(`/api/aireg/retrieval?suite=${suite}`, [suite]);
  if (loading && !data) return <Loading />;
  if (error) return <ErrorView message={error} />;
  const rows: { name: string; hit4?: number; hit5?: number; mrr?: number; extra?: string }[] = [];
  for (const [k, v] of Object.entries(data ?? {})) {
    if (v?.vector) {
      rows.push({ name: `${k} (벡터)`, hit4: v.vector.hit4, hit5: v.vector.hit5, mrr: v.vector.mrr10 });
      if (v.reranked) rows.push({ name: `${k} (+리랭커)`, hit4: v.reranked.hit4, hit5: v.reranked.hit5, mrr: v.reranked.mrr10 });
    } else if (v?.hit4 != null) {
      rows.push({ name: k, hit4: v.hit4, extra: `hit@10 ${pct(v.hit10)} / @20 ${pct(v.hit20)}` });
    }
  }
  return (
    <Card><CardContent>
      <Typography variant="subtitle2" gutterBottom>
        검색 벤치 결과 (rm_*=앵커 포함 · na_*=무앵커 · emb_*=임베더 단독)
      </Typography>
      <Table size="small">
        <TableHead><TableRow>
          <TableCell>구성</TableCell><TableCell align="right">hit@4</TableCell>
          <TableCell align="right">hit@5</TableCell><TableCell align="right">MRR@10</TableCell>
          <TableCell>비고</TableCell>
        </TableRow></TableHead>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.name}>
              <TableCell>{r.name}</TableCell>
              <TableCell align="right">{pct(r.hit4)}</TableCell>
              <TableCell align="right">{pct(r.hit5)}</TableCell>
              <TableCell align="right">{r.mrr?.toFixed(3) ?? "-"}</TableCell>
              <TableCell>{r.extra ?? ""}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </CardContent></Card>
  );
}

export default function Aireg() {
  const [tab, setTab] = useState(0);
  const [live, setLive] = useState(true);
  const [suite, setSuite] = useState("suite_v5");
  const { data: suitesData } = useApi<string[]>("/api/aireg/suites");
  return (
    <Box>
      <PageHeader title="AIReg 벤치·학습"
        subtitle="선급 규정 QA 데이터 생성 → 학습 → 판정 파이프라인 실시간 뷰"
        action={
          <Stack direction="row" spacing={2} alignItems="center">
            <TextField select size="small" label="스위트" value={suite}
              onChange={(e) => setSuite(e.target.value)} sx={{ width: 160 }}>
              {(suitesData ?? ["suite_v5"]).map((s) => <MenuItem key={s} value={s}>{s}</MenuItem>)}
            </TextField>
            <FormControlLabel control={
              <Switch checked={live} onChange={(e) => setLive(e.target.checked)} size="small" />
            } label="실시간" />
          </Stack>
        } />
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
        <Tab label="개요" /><Tab label="생성 QA" /><Tab label="평가 뷰어" />
        <Tab label="학습 실험" /><Tab label="검색 벤치" /><Tab label="로그" />
      </Tabs>
      {tab === 0 && <OverviewTab suite={suite} live={live} />}
      {tab === 1 && <QaTab suite={suite} live={live} />}
      {tab === 2 && <GenEvalTab suite={suite} live={live} />}
      {tab === 3 && <TrainTab suite={suite} />}
      {tab === 4 && <RetrievalTab suite={suite} />}
      {tab === 5 && <LogsTab suite={suite} live={live} />}
    </Box>
  );
}

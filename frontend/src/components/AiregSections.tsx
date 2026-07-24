/** AIReg 학습·평가·검색 섹션 — HPO(실험)·학습 파이프라인(로그)·성능 비교(평가)·RAG(검색 벤치)
 * 페이지에 각각 삽입되는 공용 컴포넌트. 데이터 소스는 /api/aireg/*. */
import { useState } from "react";
import {
  Box, Card, CardContent, Chip, Collapse, IconButton, MenuItem, Stack,
  Tab, Table, TableBody, TableCell, TableHead, TableRow, Tabs, TextField,
  Typography,
} from "@mui/material";
import KeyboardArrowDown from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUp from "@mui/icons-material/KeyboardArrowUp";
import { Loading, ErrorView } from "./StateViews";
import { useApi } from "../hooks/useApi";
import { useAutoRefresh } from "../hooks/useAutoRefresh";
import { AIREG_TRACKS } from "./AiregData";

const DEFAULT_SUITE = "suite_v5";

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
      </TableRow>
      <TableRow>
        <TableCell colSpan={5} sx={{ py: 0, border: 0 }}>
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

/** 실생성 평가 뷰어 — 튜닝 vs 베이스 실답변·judge 채점 (성능 비교 페이지). */
export function AiregGenEvalSection({ suite = DEFAULT_SUITE, live = true }: { suite?: string; live?: boolean }) {
  const [only, setOnly] = useState("all");
  const [track, setTrack] = useState("");
  const { data, loading, error, reload } = useApi<{ total: number; rows: GenRow[] }>(
    `/api/aireg/geneval?suite=${suite}&limit=300&only=${only}${track ? `&track=${track}` : ""}`,
    [suite, only, track]);
  useAutoRefresh(reload, 8000, live);
  return (
    <Stack spacing={2}>
      <Tabs value={track} onChange={(_, v) => setTrack(v)} variant="scrollable"
        scrollButtons="auto" sx={{ minHeight: 36, "& .MuiTab-root": { minHeight: 36, py: 0.5 } }}>
        <Tab label="전체" value="" />
        {AIREG_TRACKS.map((t) => <Tab key={t} label={t} value={t} />)}
      </Tabs>
      <Stack direction="row" spacing={2}>
        <TextField select size="small" label="필터" value={only} sx={{ width: 200 }}
          onChange={(e) => setOnly(e.target.value)}>
          <MenuItem value="all">전체</MenuItem>
          <MenuItem value="scored">채점 완료만</MenuItem>
        </TextField>
        {data && <Chip label={`${data.total}건 전체 표시`} sx={{ alignSelf: "center" }} />}
      </Stack>
      {loading && !data ? <Loading /> : error ? <ErrorView message={error} /> : (
        <Card><CardContent sx={{ p: 0 }}>
          <Table size="small">
            <TableHead><TableRow>
              <TableCell /><TableCell>트랙</TableCell><TableCell>질문</TableCell>
              <TableCell>튜닝</TableCell><TableCell>베이스</TableCell>
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

interface TrainExp { name: string; config: string; val_loss: number | null; n: number | null }

/** 32B LoRA 하이퍼파라미터 실험 결과 (HPO 페이지). */
export function AiregExperimentsSection({ suite = DEFAULT_SUITE }: { suite?: string }) {
  const { data, loading, error } = useApi<{ experiments: TrainExp[]; logs: string }>(
    `/api/aireg/train?suite=${suite}`, [suite]);
  if (loading && !data) return <Loading />;
  if (error) return <ErrorView message={error} />;
  const best = data?.experiments.find((e) => e.name !== "base32b");
  return (
    <Card><CardContent>
      <Typography variant="subtitle2" gutterBottom>
        AIReg 32B LoRA 실험 (val_loss 오름차순 · H100 · val 200)
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
  );
}

/** 본 학습(main800) 실행 로그 (학습 파이프라인 페이지). */
export function AiregTrainLogSection({ suite = DEFAULT_SUITE }: { suite?: string }) {
  const { data, loading, error } = useApi<{ experiments: TrainExp[]; logs: string }>(
    `/api/aireg/train?suite=${suite}`, [suite]);
  if (loading && !data) return <Loading />;
  if (error) return <ErrorView message={error} />;
  return (
    <Card><CardContent>
      <Typography variant="subtitle2" gutterBottom>AIReg 32B 본 학습 로그 (H100 동기화)</Typography>
      <Box component="pre" sx={{ m: 0, p: 1.5, bgcolor: "background.default", borderRadius: 1,
        fontSize: 12, maxHeight: 420, overflow: "auto", whiteSpace: "pre-wrap" }}>
        {data?.logs || "(로그 없음)"}
      </Box>
    </CardContent></Card>
  );
}

function pct(v?: number | null) { return v == null ? "-" : `${(v * 100).toFixed(1)}%`; }

/** 검색(임베더·리랭커) 벤치 결과 (RAG 지식베이스 페이지). */
export function AiregRetrievalSection({ suite = DEFAULT_SUITE }: { suite?: string }) {
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
    <Card sx={{ mb: 3 }}><CardContent>
      <Typography variant="h3" gutterBottom>검색 벤치</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        AIReg val 200 골든 근거 기준 임베더·리랭커 스택 비교 (rm_*=앵커 포함 · na_*=무앵커 · emb_*=임베더 단독)
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

/** 생성 배치 로그 (데이터셋 AIReg 생성·관리 탭). */
export function AiregBatchLogSection({ suite = DEFAULT_SUITE, live = true }: { suite?: string; live?: boolean }) {
  const { data, loading, error, reload } = useApi<{ lines: string[] }>(
    `/api/aireg/logs?suite=${suite}&name=batch&lines=120`, [suite]);
  useAutoRefresh(reload, 5000, live);
  if (loading && !data) return <Loading />;
  if (error) return <ErrorView message={error} />;
  return (
    <Card><CardContent>
      <Typography variant="subtitle2" gutterBottom>생성 배치 로그 (batch.log)</Typography>
      <Box component="pre" sx={{
        m: 0, p: 1.5, bgcolor: "background.default", borderRadius: 1,
        fontSize: 12, maxHeight: 520, overflow: "auto", whiteSpace: "pre-wrap",
      }}>
        {(data?.lines ?? []).join("\n") || "(로그 없음)"}
      </Box>
    </CardContent></Card>
  );
}

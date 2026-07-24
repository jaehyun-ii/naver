import { useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, Chip, Dialog, DialogContent, DialogTitle,
  Divider, IconButton, MenuItem, Stack, Table, TableBody, TableCell, TableHead, TableRow,
  TextField, Tooltip, Typography,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

interface CatalogItem {
  key: string;
  title: string;
  name: string;
  vars: string[];
  default: string;
  desc: string;
}

interface PromptEntry {
  name: string;
  versions: number[];
  labels: Record<string, number>;
}

interface VersionContent {
  name: string;
  version: number;
  template: string;
}

interface CompareResult {
  variant: string;
  score: number;
}

interface CompareResponse {
  results: CompareResult[];
  best: string | null;
  note: string;
}

function errMsg(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

function labelText(labels: Record<string, number>): string {
  const entries = Object.entries(labels);
  return entries.length ? entries.map(([l, v]) => `${l}→v${v}`).join(", ") : "-";
}

/** 버전 생성 폼 — 이름·템플릿 입력 후 새 버전을 추가한다. */
function CreateVersionPanel({
  name, template, onName, onTemplate, onCreated,
}: {
  name: string;
  template: string;
  onName: (v: string) => void;
  onTemplate: (v: string) => void;
  onCreated: () => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const [ok, setOk] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    setError(null);
    setOk(null);
    try {
      const r = await api<{ name: string; version: number }>("/api/prompts/version", {
        method: "POST",
        body: { name: name.trim(), template },
      });
      setOk(`${r.name} v${r.version} 생성됨`);
      onCreated();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card sx={{ flex: 1, minWidth: 320 }}>
      <CardContent>
        <Typography variant="h3" gutterBottom>버전 생성</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          불변 버전으로 프롬프트를 저장합니다. 기존 이름에 추가하면 새 버전 번호가 부여됩니다.
        </Typography>
        <TextField
          label="이름" size="small" fullWidth value={name}
          onChange={(e) => onName(e.target.value)} sx={{ mb: 2 }}
          placeholder="qa-system"
        />
        <TextField
          label="템플릿" multiline minRows={5} fullWidth value={template}
          onChange={(e) => onTemplate(e.target.value)}
          sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
        />
        <Box sx={{ mt: 2 }}>
          <Button variant="contained" onClick={submit} disabled={busy || !name.trim() || !template.trim()}>
            {busy ? "생성 중…" : "버전 추가"}
          </Button>
        </Box>
        {error && <ErrorView message={error} />}
        {ok && <Alert severity="success" sx={{ mt: 2 }}>{ok}</Alert>}
      </CardContent>
    </Card>
  );
}

/** 변형 비교 — 시스템 프롬프트 변형(한 줄에 하나)을 즉시 휴리스틱으로 채점한다. */
function ComparePanel() {
  const [text, setText] = useState(
    "간결하고 정확하게 한 문장으로 답하라.\n친절하고 상세하게 단계별로 설명하라.",
  );
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const variants = text.split("\n").map((s) => s.trim()).filter(Boolean);
      const r = await api<CompareResponse>("/api/prompts/compare", {
        method: "POST",
        body: { cases: [], variants },
      });
      setResult(r);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card sx={{ flex: 1, minWidth: 320 }}>
      <CardContent>
        <Typography variant="h3" gutterBottom>변형 비교 (즉시 휴리스틱)</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          한 줄에 하나의 프롬프트 변형을 입력하세요. 실모델 평가는 CLI(prompts.optimize)로 위임됩니다.
        </Typography>
        <TextField
          multiline minRows={5} fullWidth value={text}
          onChange={(e) => setText(e.target.value)}
          sx={{ "& textarea": { fontFamily: "ui-monospace, monospace", fontSize: 13 } }}
        />
        <Box sx={{ mt: 2 }}>
          <Button variant="contained" onClick={run} disabled={busy || !text.trim()}>
            {busy ? "비교 중…" : "비교"}
          </Button>
        </Box>
        {error && <ErrorView message={error} />}
        {result && (
          <Box sx={{ mt: 2 }}>
            {result.best && (
              <Alert severity="success" sx={{ mb: 2 }}>최우수: {result.best}</Alert>
            )}
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell align="right" sx={{ width: 96 }}>점수</TableCell>
                  <TableCell>변형</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {result.results.map((r, i) => (
                  <TableRow key={i}>
                    <TableCell align="right">{r.score.toFixed(4)}</TableCell>
                    <TableCell>{r.variant}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {result.note && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
                {result.note}
              </Typography>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

interface DiffLine { t: "same" | "add" | "del"; text: string }
function lineDiff(a: string[], b: string[]): DiffLine[] {
  const n = a.length, m = b.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out: DiffLine[] = []; let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out.push({ t: "same", text: a[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push({ t: "del", text: a[i] }); i++; }
    else { out.push({ t: "add", text: b[j] }); j++; }
  }
  while (i < n) out.push({ t: "del", text: a[i++] });
  while (j < m) out.push({ t: "add", text: b[j++] });
  return out;
}

/** 두 버전의 템플릿을 불러와 라인 단위 diff를 표시. */
function VersionDiff({ entries }: { entries: PromptEntry[] }) {
  const multi = entries.filter((e) => e.versions.length >= 2);
  const [dn, setDn] = useState("");
  const [va, setVa] = useState<number | "">("");
  const [vb, setVb] = useState<number | "">("");
  const [diff, setDiff] = useState<DiffLine[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const cur = multi.find((e) => e.name === dn);

  const run = async () => {
    if (!dn || va === "" || vb === "") return;
    setBusy(true); setErr(null); setDiff(null);
    try {
      const [a, b] = await Promise.all([
        api<VersionContent>(`/api/prompts/${encodeURIComponent(dn)}/${va}`),
        api<VersionContent>(`/api/prompts/${encodeURIComponent(dn)}/${vb}`),
      ]);
      setDiff(lineDiff(a.template.split("\n"), b.template.split("\n")));
    } catch (e) {
      setErr(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    } finally { setBusy(false); }
  };

  return (
    <Card sx={{ flex: 1, minWidth: 320 }}>
      <CardContent>
        <Typography variant="h3" gutterBottom>버전 diff</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>두 버전의 텍스트 차이(추가/삭제)를 라인 단위로 비교합니다.</Typography>
        {multi.length === 0 ? (
          <Typography variant="body2" color="text.secondary">버전이 2개 이상인 프롬프트가 없습니다.</Typography>
        ) : (
          <>
            <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: "wrap", gap: 1 }}>
              <TextField select size="small" label="프롬프트" value={dn}
                onChange={(e) => { setDn(e.target.value); setVa(""); setVb(""); setDiff(null); }} sx={{ minWidth: 150 }}>
                {multi.map((e) => <MenuItem key={e.name} value={e.name}>{e.name}</MenuItem>)}
              </TextField>
              <TextField select size="small" label="A" value={va === "" ? "" : String(va)} onChange={(e) => setVa(Number(e.target.value))} sx={{ width: 90 }} disabled={!cur}>
                {(cur?.versions ?? []).map((v) => <MenuItem key={v} value={String(v)}>v{v}</MenuItem>)}
              </TextField>
              <TextField select size="small" label="B" value={vb === "" ? "" : String(vb)} onChange={(e) => setVb(Number(e.target.value))} sx={{ width: 90 }} disabled={!cur}>
                {(cur?.versions ?? []).map((v) => <MenuItem key={v} value={String(v)}>v{v}</MenuItem>)}
              </TextField>
              <Button variant="contained" onClick={run} disabled={busy || !dn || va === "" || vb === ""}>{busy ? "…" : "비교"}</Button>
            </Stack>
            {err && <ErrorView message={err} />}
            {diff && (
              <Box sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12.5, border: 1, borderColor: "divider", borderRadius: 1, overflow: "hidden" }}>
                {diff.map((l, i) => (
                  <Box key={i} sx={{
                    px: 1, py: 0.15, whiteSpace: "pre-wrap", wordBreak: "break-word",
                    bgcolor: l.t === "add" ? "success.light" : l.t === "del" ? "error.light" : "transparent",
                  }}>
                    <Box component="span" sx={{ color: "text.disabled", mr: 1 }}>{l.t === "add" ? "+" : l.t === "del" ? "−" : " "}</Box>
                    {l.text || " "}
                  </Box>
                ))}
              </Box>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default function Prompts() {
  const catalog = useApi<CatalogItem[]>("/api/prompts/catalog");
  const list = useApi<PromptEntry[]>("/api/prompts");

  const [name, setName] = useState("qa-system");
  const [template, setTemplate] = useState("간결하고 정확하게 한 문장으로 답하라.");
  const [actionError, setActionError] = useState<string | null>(null);

  // 버전 내용 보기 다이얼로그
  const [viewing, setViewing] = useState<VersionContent | null>(null);
  const [viewLoading, setViewLoading] = useState(false);

  const reloadAll = () => {
    catalog.reload();
    list.reload();
  };

  const promote = async (n: string, version: number) => {
    setActionError(null);
    try {
      await api("/api/prompts/promote", {
        method: "POST",
        body: { name: n, version, label: "prod" },
      });
      list.reload();
    } catch (e) {
      setActionError(errMsg(e));
    }
  };

  const view = async (n: string, version: number) => {
    setActionError(null);
    setViewLoading(true);
    setViewing({ name: n, version, template: "" });
    try {
      const r = await api<VersionContent>(`/api/prompts/${encodeURIComponent(n)}/${version}`);
      setViewing(r);
    } catch (e) {
      setActionError(errMsg(e));
      setViewing(null);
    } finally {
      setViewLoading(false);
    }
  };

  const editFromCatalog = (c: CatalogItem) => {
    setName(c.name);
    setTemplate(c.default);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const entries = list.data ?? [];

  return (
    <>
      <PageHeader
        title="프롬프트"
        subtitle="프롬프트 카탈로그·버전 관리 — 불변 버전 · 라벨(prod/stg) · 롤백 · 변형 비교"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={reloadAll} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      {actionError && <ErrorView message={actionError} />}

      <Stack direction="row" spacing={3} sx={{ mb: 3, flexWrap: "wrap", gap: 3 }}>
        <CreateVersionPanel
          name={name} template={template}
          onName={setName} onTemplate={setTemplate}
          onCreated={() => list.reload()}
        />
        <ComparePanel />
        <VersionDiff entries={entries} />
      </Stack>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>시스템 특수 프롬프트</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            가드레일·RAG 프롬프트도 버전 자산으로 관리됩니다. "편집"으로 기본값을 위 버전 생성 폼에 불러와
            수정 후 prod 승격하면 코드 수정 없이 반영됩니다.
          </Typography>
          {catalog.loading && !catalog.data ? (
            <Loading />
          ) : catalog.error ? (
            <ErrorView message={catalog.error} />
          ) : !catalog.data || catalog.data.length === 0 ? (
            <EmptyView message="표시할 카탈로그 항목이 없습니다." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>용도</TableCell>
                  <TableCell>프롬프트 이름</TableCell>
                  <TableCell>변수</TableCell>
                  <TableCell>현재 라벨</TableCell>
                  <TableCell align="right"></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {catalog.data.map((c) => {
                  const reg = entries.find((p) => p.name === c.name);
                  const hasLabels = reg && Object.keys(reg.labels).length > 0;
                  return (
                    <TableRow key={c.key}>
                      <TableCell>
                        <Typography variant="body2" fontWeight={600}>{c.title}</Typography>
                        <Typography variant="caption" color="text.secondary">{c.desc}</Typography>
                      </TableCell>
                      <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{c.name}</TableCell>
                      <TableCell>
                        <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", gap: 0.5 }}>
                          {c.vars.map((v) => (
                            <Chip key={v} size="small" variant="outlined" label={`{${v}}`} />
                          ))}
                        </Stack>
                      </TableCell>
                      <TableCell>
                        {hasLabels ? (
                          <Typography variant="body2">{labelText(reg!.labels)}</Typography>
                        ) : (
                          <Typography variant="caption" color="text.secondary">미등록(기본값 사용)</Typography>
                        )}
                      </TableCell>
                      <TableCell align="right">
                        <Button size="small" variant="outlined" onClick={() => editFromCatalog(c)}>편집</Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>프롬프트 레지스트리</Typography>
          {list.loading && !list.data ? (
            <Loading />
          ) : list.error ? (
            <ErrorView message={list.error} />
          ) : entries.length === 0 ? (
            <EmptyView message="아직 등록된 프롬프트가 없습니다. 위에서 버전을 생성하세요." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>이름</TableCell>
                  <TableCell>버전</TableCell>
                  <TableCell>라벨</TableCell>
                  <TableCell align="right">보기 · 승격</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {entries.map((p) => (
                  <TableRow key={p.name}>
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{p.name}</TableCell>
                    <TableCell>{p.versions.join(", ") || "-"}</TableCell>
                    <TableCell>{labelText(p.labels)}</TableCell>
                    <TableCell align="right">
                      <Stack direction="row" spacing={0.5} justifyContent="flex-end" sx={{ flexWrap: "wrap", gap: 0.5 }}>
                        {p.versions.map((v) => (
                          <Button key={`view-${v}`} size="small" variant="text" onClick={() => view(p.name, v)}>
                            v{v} 보기
                          </Button>
                        ))}
                        {p.versions.map((v) => (
                          <Button key={`promote-${v}`} size="small" variant="outlined" onClick={() => promote(p.name, v)}>
                            v{v}→prod
                          </Button>
                        ))}
                      </Stack>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Dialog open={viewing !== null} onClose={() => setViewing(null)} fullWidth maxWidth="md">
        <DialogTitle>
          {viewing ? `${viewing.name} · v${viewing.version}` : "버전 내용"}
        </DialogTitle>
        <Divider />
        <DialogContent>
          {viewLoading ? (
            <Loading />
          ) : (
            <Box
              component="pre"
              sx={{
                m: 0, p: 2, borderRadius: 1, bgcolor: "action.hover",
                fontFamily: "ui-monospace, monospace", fontSize: 13,
                whiteSpace: "pre-wrap", wordBreak: "break-word",
              }}
            >
              {viewing?.template}
            </Box>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

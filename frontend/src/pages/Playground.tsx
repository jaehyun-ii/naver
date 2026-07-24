import { useEffect, useRef, useState } from "react";
import {
  Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Card, CardContent,
  Chip, CircularProgress, FormControl, IconButton, InputLabel, MenuItem, Paper, Select,
  Slider, Stack, TextField, ToggleButton, ToggleButtonGroup, Tooltip, Typography,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import ChatOutlined from "@mui/icons-material/ChatOutlined";
import { PageHeader } from "../components/PageHeader";
import { ErrorView } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

interface Models { models: string[] }
type Role = "user" | "assistant" | "system";
interface Message { role: Role; content: string }
interface ChatReply {
  content: string;
  model: string;
  backend: string;
  usage: Record<string, number>;
  cost_usd: number;
}
interface CompareResult { model: string; reply?: ChatReply; error?: string; ms: number }

const VKEY_STORAGE = "llmops.virtualKey";

function Bubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <Box sx={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start" }}>
      <Paper variant="outlined" sx={{
        px: 1.5, py: 1, maxWidth: "80%", whiteSpace: "pre-wrap", wordBreak: "break-word",
        bgcolor: isUser ? "primary.light" : "background.paper",
        borderColor: isUser ? "primary.light" : "divider",
      }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.25 }}>{msg.role}</Typography>
        <Typography variant="body2">{msg.content}</Typography>
      </Paper>
    </Box>
  );
}

export default function Playground() {
  const { data: modelsData, error: modelsError } = useApi<Models>("/api/models");
  const models = modelsData?.models ?? [];

  const [mode, setMode] = useState<"chat" | "compare">("chat");
  const [model, setModel] = useState("");
  const [vkey, setVkey] = useState(() => localStorage.getItem(VKEY_STORAGE) ?? "");
  const [input, setInput] = useState("안녕하세요");
  const [msgs, setMsgs] = useState<Message[]>([]);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(256);
  const [error, setError] = useState<string | null>(null);
  const [last, setLast] = useState<(ChatReply & { ms: number }) | null>(null);
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  // 비교 모드
  const [cmpModels, setCmpModels] = useState<string[]>([]);
  const [cmpInput, setCmpInput] = useState("선급증서 유효기간이 지나면 어떻게 하나요?");
  const [cmpResults, setCmpResults] = useState<CompareResult[] | null>(null);
  const [cmpBusy, setCmpBusy] = useState(false);

  useEffect(() => { if (!model && models.length) setModel(models[0]); }, [models, model]);
  useEffect(() => { if (!cmpModels.length && models.length) setCmpModels(models.slice(0, 2)); }, [models, cmpModels.length]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);

  const send = async () => {
    const text = input.trim();
    if (!text || !vkey || !model || busy) return;
    setError(null);
    const next: Message[] = [...msgs, { role: "user", content: text }];
    setMsgs(next); setInput(""); setBusy(true);
    const t0 = performance.now();
    try {
      localStorage.setItem(VKEY_STORAGE, vkey);
      const r = await api<ChatReply>("/api/chat", {
        method: "POST",
        body: { virtual_key: vkey, model, messages: next, temperature, max_tokens: maxTokens },
      });
      setMsgs([...next, { role: "assistant", content: r.content }]);
      setLast({ ...r, ms: Math.round(performance.now() - t0) });
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    } finally { setBusy(false); }
  };

  const runCompare = async () => {
    const text = cmpInput.trim();
    if (!text || !vkey || cmpModels.length === 0 || cmpBusy) return;
    setError(null); setCmpBusy(true); setCmpResults(null);
    localStorage.setItem(VKEY_STORAGE, vkey);
    const messages: Message[] = [{ role: "user", content: text }];
    const results = await Promise.all(cmpModels.map(async (m): Promise<CompareResult> => {
      const t0 = performance.now();
      try {
        const r = await api<ChatReply>("/api/chat", {
          method: "POST", body: { virtual_key: vkey, model: m, messages, temperature, max_tokens: maxTokens },
        });
        return { model: m, reply: r, ms: Math.round(performance.now() - t0) };
      } catch (e) {
        return { model: m, error: e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message, ms: Math.round(performance.now() - t0) };
      }
    }));
    setCmpResults(results); setCmpBusy(false);
  };

  const clear = () => { setMsgs([]); setLast(null); setError(null); setCmpResults(null); };
  const fastest = cmpResults?.filter((r) => !r.error).reduce<CompareResult | null>((a, r) => (!a || r.ms < a.ms ? r : a), null);

  return (
    <>
      <PageHeader
        title="플레이그라운드"
        subtitle="가상키 인증 → 정책(화이트리스트·RPM·예산) 적용 → 게이트웨이 라우팅. 여러 모델 응답을 나란히 비교."
        action={
          <Tooltip title="대화·결과 지우기"><span>
            <IconButton onClick={clear} disabled={!msgs.length && !cmpResults} aria-label="지우기"><DeleteSweepIcon /></IconButton>
          </span></Tooltip>
        }
      />

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} sx={{ mb: 2 }} alignItems={{ md: "center" }}>
            <ToggleButtonGroup exclusive size="small" value={mode} onChange={(_, v) => v && setMode(v)}>
              <ToggleButton value="chat"><ChatOutlined fontSize="small" sx={{ mr: 0.5 }} />대화</ToggleButton>
              <ToggleButton value="compare"><CompareArrowsIcon fontSize="small" sx={{ mr: 0.5 }} />모델 비교</ToggleButton>
            </ToggleButtonGroup>
            <TextField
              label="가상키 (sk-...)" value={vkey} onChange={(e) => setVkey(e.target.value)}
              size="small" type="password" sx={{ flex: 1, minWidth: 220 }}
            />
          </Stack>

          <Accordion elevation={0} disableGutters sx={{ mb: 2, border: 1, borderColor: "divider", "&:before": { display: "none" } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="body2" color="text.secondary">생성 파라미터 · temperature {temperature.toFixed(2)} · max_tokens {maxTokens}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={3} sx={{ px: 1 }}>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="caption" color="text.secondary">temperature: {temperature.toFixed(2)}</Typography>
                  <Slider value={temperature} min={0} max={2} step={0.05} onChange={(_, v) => setTemperature(v as number)} valueLabelDisplay="auto" size="small" />
                </Box>
                <TextField label="max_tokens" type="number" size="small" value={maxTokens}
                  onChange={(e) => setMaxTokens(Math.max(1, Number(e.target.value) || 1))} sx={{ width: 140 }} />
              </Stack>
            </AccordionDetails>
          </Accordion>

          {mode === "chat" ? (
            <>
              <FormControl size="small" sx={{ minWidth: 260, mb: 2 }}>
                <InputLabel id="model-label">모델</InputLabel>
                <Select labelId="model-label" label="모델" value={model} onChange={(e) => setModel(e.target.value)} displayEmpty>
                  {models.length === 0 && <MenuItem value="" disabled>(모델 로드: 마스터 키 필요)</MenuItem>}
                  {models.map((m) => <MenuItem key={m} value={m}>{m}</MenuItem>)}
                </Select>
              </FormControl>

              <Box sx={{ minHeight: 240, maxHeight: 440, overflowY: "auto", p: 1.5, mb: 2, bgcolor: "background.default", border: 1, borderColor: "divider", borderRadius: 1 }}>
                {msgs.length === 0 ? (
                  <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: "center" }}>가상키와 모델을 선택하고 메시지를 보내보세요.</Typography>
                ) : (
                  <Stack spacing={1.5}>{msgs.map((m, i) => <Bubble key={i} msg={m} />)}</Stack>
                )}
                <div ref={endRef} />
              </Box>

              {error && <ErrorView message={error} />}

              <Stack direction="row" spacing={1}>
                <TextField fullWidth size="small" placeholder="메시지를 입력하세요" value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} disabled={busy} />
                <Button variant="contained" endIcon={<SendIcon />} onClick={send} disabled={!vkey || !model || !input.trim() || busy}>
                  {busy ? "전송 중" : "보내기"}
                </Button>
              </Stack>

              {last && (
                <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap", gap: 1 }}>
                  <Chip size="small" variant="outlined" label={`backend: ${last.backend}`} />
                  <Chip size="small" variant="outlined" label={`model: ${last.model}`} />
                  <Chip size="small" variant="outlined" label={`tokens: ${last.usage.total_tokens ?? 0}`} />
                  <Chip size="small" variant="outlined" label={`cost: $${last.cost_usd}`} />
                  <Chip size="small" variant="outlined" label={`지연: ${last.ms}ms`} />
                </Stack>
              )}
            </>
          ) : (
            <>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>비교할 모델 선택 ({cmpModels.length}개)</Typography>
              <Stack direction="row" flexWrap="wrap" useFlexGap spacing={1} sx={{ mb: 2 }}>
                {models.length === 0 && <Typography variant="caption" color="text.secondary">(모델 로드: 마스터 키 필요)</Typography>}
                {models.map((m) => {
                  const on = cmpModels.includes(m);
                  return (
                    <Chip key={m} label={m} clickable color={on ? "primary" : "default"} variant={on ? "filled" : "outlined"}
                      onClick={() => setCmpModels((s) => on ? s.filter((x) => x !== m) : [...s, m])} />
                  );
                })}
              </Stack>

              <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                <TextField fullWidth size="small" placeholder="동일 입력으로 여러 모델을 비교합니다" value={cmpInput}
                  onChange={(e) => setCmpInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runCompare(); } }} disabled={cmpBusy} />
                <Button variant="contained" startIcon={cmpBusy ? <CircularProgress size={16} color="inherit" /> : <CompareArrowsIcon />}
                  onClick={runCompare} disabled={!vkey || cmpModels.length === 0 || !cmpInput.trim() || cmpBusy}>
                  {cmpBusy ? "실행 중" : "비교 실행"}
                </Button>
              </Stack>

              {error && <ErrorView message={error} />}

              {cmpResults && (
                <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", lg: `repeat(${Math.min(cmpResults.length, 3)},1fr)` } }}>
                  {cmpResults.map((res) => (
                    <Card key={res.model} variant="outlined" sx={{ borderRadius: 2, borderColor: res === fastest ? "success.main" : "divider" }}>
                      <CardContent>
                        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                          <Typography variant="body2" fontWeight={700} noWrap>{res.model}</Typography>
                          <Chip size="small" color={res === fastest ? "success" : "default"} variant={res === fastest ? "filled" : "outlined"} label={`${res.ms}ms${res === fastest ? " · 최속" : ""}`} />
                        </Stack>
                        {res.error ? (
                          <Alert severity="error" sx={{ fontSize: 13 }}>{res.error}</Alert>
                        ) : (
                          <>
                            <Typography variant="body2" sx={{ whiteSpace: "pre-wrap", wordBreak: "break-word", minHeight: 72 }}>{res.reply?.content}</Typography>
                            <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mt: 1.5 }}>
                              <Chip size="small" variant="outlined" label={`backend ${res.reply?.backend}`} />
                              <Chip size="small" variant="outlined" label={`tokens ${res.reply?.usage.total_tokens ?? 0}`} />
                              <Chip size="small" variant="outlined" label={`$${res.reply?.cost_usd}`} />
                            </Stack>
                          </>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              )}
            </>
          )}

          {modelsError && (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 2 }}>
              모델 목록을 불러오지 못했습니다 (우상단 열쇠 아이콘에서 마스터 키 설정): {modelsError}
            </Typography>
          )}
        </CardContent>
      </Card>
    </>
  );
}

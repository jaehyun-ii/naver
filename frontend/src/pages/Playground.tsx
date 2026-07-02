import { useEffect, useRef, useState } from "react";
import {
  Accordion, AccordionDetails, AccordionSummary, Box, Button, Card, CardContent,
  Chip, FormControl, IconButton, InputLabel, MenuItem, Paper, Select, Slider,
  Stack, TextField, Tooltip, Typography,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
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

const VKEY_STORAGE = "llmops.virtualKey";

function Bubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <Box sx={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start" }}>
      <Paper
        variant="outlined"
        sx={{
          px: 1.5, py: 1, maxWidth: "80%", whiteSpace: "pre-wrap", wordBreak: "break-word",
          bgcolor: isUser ? "primary.light" : "background.paper",
          borderColor: isUser ? "primary.light" : "divider",
        }}
      >
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.25 }}>
          {msg.role}
        </Typography>
        <Typography variant="body2">{msg.content}</Typography>
      </Paper>
    </Box>
  );
}

export default function Playground() {
  const { data: modelsData, error: modelsError } = useApi<Models>("/api/models");
  const models = modelsData?.models ?? [];

  const [model, setModel] = useState("");
  const [vkey, setVkey] = useState(() => localStorage.getItem(VKEY_STORAGE) ?? "");
  const [input, setInput] = useState("안녕하세요");
  const [msgs, setMsgs] = useState<Message[]>([]);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(256);
  const [error, setError] = useState<string | null>(null);
  const [last, setLast] = useState<ChatReply | null>(null);
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  // 모델 목록이 로드되면 첫 모델을 기본 선택
  useEffect(() => {
    if (!model && models.length) setModel(models[0]);
  }, [models, model]);

  // 새 메시지 시 하단으로 스크롤
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  const send = async () => {
    const text = input.trim();
    if (!text || !vkey || !model || busy) return;
    setError(null);
    const next: Message[] = [...msgs, { role: "user", content: text }];
    setMsgs(next);
    setInput("");
    setBusy(true);
    try {
      localStorage.setItem(VKEY_STORAGE, vkey);
      const r = await api<ChatReply>("/api/chat", {
        method: "POST",
        body: { virtual_key: vkey, model, messages: next, temperature, max_tokens: maxTokens },
      });
      setMsgs([...next, { role: "assistant", content: r.content }]);
      setLast(r);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const clear = () => { setMsgs([]); setLast(null); setError(null); };

  return (
    <>
      <PageHeader
        title="챗 플레이그라운드"
        subtitle="가상키 인증 → 정책(화이트리스트·RPM·예산) 적용 → 백엔드 호출. 모델 미연결 시 echo 백엔드로 동작."
        action={
          <Tooltip title="대화 지우기">
            <span>
              <IconButton onClick={clear} disabled={!msgs.length} aria-label="대화 지우기">
                <DeleteSweepIcon />
              </IconButton>
            </span>
          </Tooltip>
        }
      />

      <Card>
        <CardContent>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mb: 2 }}>
            <TextField
              label="가상키 (sk-...)" value={vkey} onChange={(e) => setVkey(e.target.value)}
              size="small" type="password" sx={{ flex: 1, minWidth: 220 }}
            />
            <FormControl size="small" sx={{ minWidth: 220 }}>
              <InputLabel id="model-label">모델</InputLabel>
              <Select
                labelId="model-label" label="모델" value={model}
                onChange={(e) => setModel(e.target.value)}
                displayEmpty
              >
                {models.length === 0 && (
                  <MenuItem value="" disabled>
                    (모델 로드: 마스터 키 필요)
                  </MenuItem>
                )}
                {models.map((m) => (
                  <MenuItem key={m} value={m}>{m}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Stack>

          <Accordion elevation={0} disableGutters sx={{ mb: 2, border: 1, borderColor: "divider", "&:before": { display: "none" } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="body2" color="text.secondary">고급 설정 (temperature · max_tokens)</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={3} sx={{ px: 1 }}>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="caption" color="text.secondary">temperature: {temperature.toFixed(2)}</Typography>
                  <Slider
                    value={temperature} min={0} max={2} step={0.05}
                    onChange={(_, v) => setTemperature(v as number)}
                    valueLabelDisplay="auto" size="small"
                  />
                </Box>
                <TextField
                  label="max_tokens" type="number" size="small" value={maxTokens}
                  onChange={(e) => setMaxTokens(Math.max(1, Number(e.target.value) || 1))}
                  sx={{ width: 140 }}
                />
              </Stack>
            </AccordionDetails>
          </Accordion>

          <Box
            sx={{
              minHeight: 240, maxHeight: 440, overflowY: "auto", p: 1.5, mb: 2,
              bgcolor: "background.default", border: 1, borderColor: "divider", borderRadius: 1,
            }}
          >
            {msgs.length === 0 ? (
              <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: "center" }}>
                가상키와 모델을 선택하고 메시지를 보내보세요.
              </Typography>
            ) : (
              <Stack spacing={1.5}>
                {msgs.map((m, i) => <Bubble key={i} msg={m} />)}
              </Stack>
            )}
            <div ref={endRef} />
          </Box>

          {error && <ErrorView message={error} />}

          <Stack direction="row" spacing={1}>
            <TextField
              fullWidth size="small" placeholder="메시지를 입력하세요" value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              disabled={busy}
            />
            <Button
              variant="contained" endIcon={<SendIcon />} onClick={send}
              disabled={!vkey || !model || !input.trim() || busy}
            >
              {busy ? "전송 중" : "보내기"}
            </Button>
          </Stack>

          {last && (
            <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap", gap: 1 }}>
              <Chip size="small" variant="outlined" label={`backend: ${last.backend}`} />
              <Chip size="small" variant="outlined" label={`model: ${last.model}`} />
              <Chip size="small" variant="outlined" label={`tokens: ${last.usage.total_tokens ?? 0}`} />
              <Chip size="small" variant="outlined" label={`cost: $${last.cost_usd}`} />
            </Stack>
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

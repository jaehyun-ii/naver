import { useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, Chip, LinearProgress, Stack,
  TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

const DEFAULT_NAME = "hcx-seed-tuned";

interface RolloutTrack {
  api_base: string;
  weight: number;
}

interface Rollout {
  name: string;
  stable: RolloutTrack | null;
  canary: RolloutTrack | null;
}

function TrackBar({
  label, track, color,
}: {
  label: string;
  track: RolloutTrack;
  color: "success" | "warning";
}) {
  const weight = Math.max(0, Math.min(100, track.weight));
  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", mb: 0.5, gap: 1 }}>
        <Stack direction="row" spacing={1} alignItems="center">
          <Chip size="small" label={label} color={color} variant="outlined" />
          <Typography variant="body2" fontWeight={600}>{track.weight}%</Typography>
        </Stack>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontFamily: "ui-monospace, monospace", overflow: "hidden", textOverflow: "ellipsis" }}
        >
          {track.api_base}
        </Typography>
      </Box>
      <LinearProgress variant="determinate" value={weight} color={color} />
    </Box>
  );
}

export default function Serving() {
  const [nameInput, setNameInput] = useState(DEFAULT_NAME);
  const [activeName, setActiveName] = useState(DEFAULT_NAME);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const { data, loading, error, reload } = useApi<Rollout>(
    `/api/serving/rollout/${encodeURIComponent(activeName)}`,
    [activeName],
  );

  const query = () => {
    const next = nameInput.trim();
    if (!next) return;
    setActionError(null);
    setNotice(null);
    if (next === activeName) reload();
    else setActiveName(next);
  };

  const act = async (kind: "promote" | "rollback") => {
    setBusy(true);
    setActionError(null);
    setNotice(null);
    try {
      await api(`/api/serving/${encodeURIComponent(activeName)}/${kind}`, { method: "POST" });
      setNotice(kind === "promote"
        ? `카나리를 stable로 승격했습니다 · ${activeName}`
        : `카나리를 롤백했습니다 (stable 100% 복귀) · ${activeName}`);
      reload();
    } catch (e) {
      const msg = e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
      setActionError(msg);
    } finally {
      setBusy(false);
    }
  };

  const hasCanary = Boolean(data?.canary);

  return (
    <>
      <PageHeader
        title="서빙 · 카나리"
        subtitle="신모델을 소량(weight%) 노출 → 지표 확인 → 승격(100%) 또는 롤백. 게이트웨이 가중 라우팅."
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>롤아웃 조회</Typography>
          <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap", gap: 2, alignItems: "center" }}>
            <TextField
              label="논리 모델명"
              size="small"
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") query(); }}
              sx={{ minWidth: 240 }}
            />
            <Button variant="outlined" onClick={query} disabled={!nameInput.trim()}>
              롤아웃 조회
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {notice && <Alert severity="success" sx={{ mb: 2 }}>{notice}</Alert>}
      {actionError && <ErrorView message={actionError} />}

      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>트래픽 분배 · {activeName}</Typography>
          {loading && !data ? (
            <Loading label="롤아웃 상태 조회 중…" />
          ) : error ? (
            <ErrorView message={error} />
          ) : !data || (!data.stable && !data.canary) ? (
            <EmptyView message="해당 논리 모델명의 롤아웃 정보가 없습니다. 모델명을 확인하세요." />
          ) : (
            <Stack spacing={2.5}>
              {data.stable
                ? <TrackBar label="stable" track={data.stable} color="success" />
                : <EmptyView message="stable 트랙이 없습니다." />}
              {data.canary
                ? <TrackBar label="canary" track={data.canary} color="warning" />
                : (
                  <Typography variant="body2" color="text.secondary">
                    카나리 없음 (단일 stable)
                  </Typography>
                )}

              <Stack direction="row" spacing={1.5} sx={{ pt: 1, flexWrap: "wrap", gap: 1.5 }}>
                <Button
                  variant="contained"
                  color="success"
                  disabled={busy || !hasCanary}
                  onClick={() => act("promote")}
                >
                  {busy ? "처리 중…" : "승격(promote) → 100%"}
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  disabled={busy || !hasCanary}
                  onClick={() => act("rollback")}
                >
                  롤백(rollback) → stable
                </Button>
              </Stack>
              {!hasCanary && (
                <Typography variant="caption" color="text.secondary">
                  승격·롤백은 카나리가 존재할 때만 가능합니다. 카나리 배포는 파이프라인의 배포 단계 또는
                  executor.deploy_canary로 생성됩니다(다중 GPU 필요).
                </Typography>
              )}
            </Stack>
          )}
        </CardContent>
      </Card>
    </>
  );
}

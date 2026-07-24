import { useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, Chip, Dialog, DialogActions,
  DialogContent, DialogContentText, DialogTitle, IconButton, InputAdornment,
  LinearProgress, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, Tooltip,
  Typography,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import VpnKeyOutlined from "@mui/icons-material/VpnKeyOutlined";
import GroupsOutlined from "@mui/icons-material/GroupsOutlined";
import PaidOutlined from "@mui/icons-material/PaidOutlined";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { SummaryTiles } from "../components/SummaryTiles";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

interface KeyView {
  key_id: string;
  tenant_id: string;
  allowed_models: string[];
  monthly_budget_usd?: number | null;
  rpm_limit?: number | null;
  spent_usd: number;
}

interface IssuedKey extends KeyView {
  virtual_key: string;
}

interface IssueKeyBody {
  tenant_id: string;
  allowed_models: string[];
  monthly_budget_usd?: number | null;
  rpm_limit?: number | null;
}

interface FormState {
  tenant_id: string;
  allowed_models: string;
  monthly_budget_usd: string;
  rpm_limit: string;
}

const EMPTY_FORM: FormState = {
  tenant_id: "",
  allowed_models: "",
  monthly_budget_usd: "",
  rpm_limit: "",
};

function errMsg(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

export default function Keys() {
  const { data, loading, error, reload } = useApi<KeyView[]>("/api/keys");
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [issuing, setIssuing] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [issued, setIssued] = useState<IssuedKey | null>(null);
  const [copied, setCopied] = useState(false);

  const [revokeTarget, setRevokeTarget] = useState<KeyView | null>(null);
  const [revoking, setRevoking] = useState(false);
  const [revokeError, setRevokeError] = useState<string | null>(null);

  const set = (patch: Partial<FormState>) => setForm((f) => ({ ...f, ...patch }));

  const issue = async () => {
    setIssuing(true);
    setFormError(null);
    try {
      const models = form.allowed_models
        .split(",")
        .map((m) => m.trim())
        .filter(Boolean);
      const body: IssueKeyBody = {
        tenant_id: form.tenant_id.trim(),
        allowed_models: models,
        monthly_budget_usd: form.monthly_budget_usd ? Number(form.monthly_budget_usd) : null,
        rpm_limit: form.rpm_limit ? Number(form.rpm_limit) : null,
      };
      if (!body.tenant_id) throw new Error("테넌트 ID를 입력하세요.");
      const r = await api<IssuedKey>("/api/keys", { method: "POST", body });
      setIssued(r);
      setCopied(false);
      setForm(EMPTY_FORM);
      reload();
    } catch (e) {
      setFormError(errMsg(e));
    } finally {
      setIssuing(false);
    }
  };

  const revoke = async () => {
    if (!revokeTarget) return;
    setRevoking(true);
    setRevokeError(null);
    try {
      await api(`/api/keys/${revokeTarget.key_id}`, { method: "DELETE" });
      setRevokeTarget(null);
      reload();
    } catch (e) {
      setRevokeError(errMsg(e));
    } finally {
      setRevoking(false);
    }
  };

  const copySecret = async () => {
    if (!issued) return;
    try {
      await navigator.clipboard.writeText(issued.virtual_key);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <>
      <PageHeader
        title="테넌트 키"
        subtitle="테넌트별 가상키 발급, 모델 화이트리스트, 월 예산·RPM 제한, 사용량 조회"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      {(data?.length ?? 0) > 0 && (
        <SummaryTiles stats={[
          { label: "발급 키", value: data!.length, hint: "활성 가상키", icon: VpnKeyOutlined },
          { label: "테넌트", value: new Set(data!.map((k) => k.tenant_id)).size, icon: GroupsOutlined },
          { label: "총 사용액", value: `$${data!.reduce((a, k) => a + (k.spent_usd ?? 0), 0).toFixed(2)}`, hint: "월 누적", icon: PaidOutlined, accent: "success.main" },
        ]} />
      )}

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>가상키 발급</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            발급된 평문 키는 <strong>발급 직후 1회만</strong> 표시됩니다. 안전한 곳에 즉시 보관하세요.
          </Typography>
          <Stack spacing={2}>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <TextField
                label="테넌트 ID" required fullWidth size="small"
                value={form.tenant_id}
                onChange={(e) => set({ tenant_id: e.target.value })}
              />
              <TextField
                label="허용 모델 (쉼표로 구분)" fullWidth size="small"
                placeholder="비우면 전체 허용"
                value={form.allowed_models}
                onChange={(e) => set({ allowed_models: e.target.value })}
              />
            </Stack>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <TextField
                label="월 예산 USD" fullWidth size="small" type="number"
                placeholder="비우면 무제한"
                value={form.monthly_budget_usd}
                onChange={(e) => set({ monthly_budget_usd: e.target.value })}
              />
              <TextField
                label="RPM 제한" fullWidth size="small" type="number"
                placeholder="비우면 무제한"
                value={form.rpm_limit}
                onChange={(e) => set({ rpm_limit: e.target.value })}
              />
            </Stack>
          </Stack>
          <Box sx={{ mt: 2 }}>
            <Button variant="contained" onClick={issue} disabled={issuing || !form.tenant_id.trim()}>
              {issuing ? "발급 중…" : "가상키 발급"}
            </Button>
          </Box>

          {formError && <ErrorView message={formError} />}

          {issued && (
            <Alert
              severity="success"
              sx={{ mt: 2 }}
              onClose={() => setIssued(null)}
              action={
                <Tooltip title={copied ? "복사됨" : "복사"}>
                  <IconButton size="small" onClick={copySecret} aria-label="가상키 복사">
                    <ContentCopyIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              }
            >
              <Typography variant="body2" fontWeight={600} gutterBottom>
                가상키 발급 완료 — 이 값은 다시 표시되지 않습니다.
              </Typography>
              <TextField
                fullWidth size="small" value={issued.virtual_key}
                InputProps={{
                  readOnly: true,
                  sx: { fontFamily: "ui-monospace, monospace", fontSize: 13 },
                  startAdornment: (
                    <InputAdornment position="start">
                      <Chip size="small" variant="outlined" label={issued.tenant_id} />
                    </InputAdornment>
                  ),
                }}
                onFocus={(e) => e.target.select()}
              />
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>발급된 가상키</Typography>
          {loading && !data ? (
            <Loading />
          ) : error ? (
            <ErrorView message={error} />
          ) : !data || data.length === 0 ? (
            <EmptyView message="아직 발급된 가상키가 없습니다. 위에서 발급하세요." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>key_id</TableCell>
                  <TableCell>테넌트</TableCell>
                  <TableCell>허용 모델</TableCell>
                  <TableCell sx={{ minWidth: 190 }}>예산 사용</TableCell>
                  <TableCell align="right">RPM</TableCell>
                  <TableCell align="right">액션</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((k) => (
                  <TableRow key={k.key_id}>
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                      {k.key_id}
                    </TableCell>
                    <TableCell>{k.tenant_id}</TableCell>
                    <TableCell>
                      {k.allowed_models.length === 0 ? (
                        <Typography variant="caption" color="text.secondary">전체 허용</Typography>
                      ) : (
                        <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", gap: 0.5 }}>
                          {k.allowed_models.map((m) => (
                            <Chip key={m} size="small" variant="outlined" label={m} />
                          ))}
                        </Stack>
                      )}
                    </TableCell>
                    <TableCell>
                      {k.monthly_budget_usd == null ? (
                        <Typography variant="caption" color="text.secondary">무제한 · ${k.spent_usd.toFixed(2)} 사용</Typography>
                      ) : (() => {
                        const pct = Math.min(100, (k.spent_usd / (k.monthly_budget_usd || 1)) * 100);
                        const color: "error" | "warning" | "success" = pct >= 90 ? "error" : pct >= 70 ? "warning" : "success";
                        return (
                          <Box sx={{ minWidth: 170 }}>
                            <Stack direction="row" justifyContent="space-between" alignItems="baseline">
                              <Typography variant="caption" sx={{ fontFamily: "ui-monospace, monospace" }}>
                                ${k.spent_usd.toFixed(2)} / ${k.monthly_budget_usd.toFixed(2)}
                              </Typography>
                              <Typography variant="caption" color={`${color}.main`} fontWeight={700}>{Math.round(pct)}%</Typography>
                            </Stack>
                            <LinearProgress variant="determinate" value={pct} color={color} sx={{ height: 5, borderRadius: 2, mt: 0.25 }} />
                          </Box>
                        );
                      })()}
                    </TableCell>
                    <TableCell align="right">{k.rpm_limit ?? "무제한"}</TableCell>
                    <TableCell align="right">
                      <Button
                        size="small" variant="outlined" color="error"
                        onClick={() => { setRevokeTarget(k); setRevokeError(null); }}
                      >
                        폐기
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Dialog open={!!revokeTarget} onClose={() => setRevokeTarget(null)} fullWidth maxWidth="xs">
        <DialogTitle>가상키 폐기</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 1, fontSize: 14 }}>
            테넌트 <strong>{revokeTarget?.tenant_id}</strong>의 가상키{" "}
            <code>{revokeTarget?.key_id}</code>를 폐기합니다. 이 작업은 되돌릴 수 없으며,
            해당 키로의 호출은 즉시 차단됩니다.
          </DialogContentText>
          {revokeError && <ErrorView message={revokeError} />}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRevokeTarget(null)}>취소</Button>
          <Button variant="contained" color="error" onClick={revoke} disabled={revoking}>
            {revoking ? "폐기 중…" : "폐기"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

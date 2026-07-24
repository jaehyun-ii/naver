import { useState } from "react";
import {
  Alert, Box, Button, Card, CardContent, Chip, IconButton, InputAdornment, Stack,
  Tab, Table, TableBody, TableCell, TableHead, TableRow, Tabs, TextField, Tooltip,
  Typography,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import SecurityOutlined from "@mui/icons-material/SecurityOutlined";
import VpnKeyOutlined from "@mui/icons-material/VpnKeyOutlined";
import HistoryOutlined from "@mui/icons-material/HistoryOutlined";
import CheckOutlined from "@mui/icons-material/Check";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { SummaryTiles } from "../components/SummaryTiles";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

interface RolesResponse {
  roles: string[];
}

interface IssuedToken {
  token_id: string;
  subject: string;
  roles: string[];
}

interface IssueTokenResponse {
  token: string;
  subject: string;
  roles: string[];
  note?: string;
}

interface AuditEntry {
  ts: number | string;
  actor: string;
  action: string;
  target?: string | null;
  result: string;
  detail?: Record<string, unknown>;
}

function toMessage(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

function resultLabel(result: string): string {
  const r = result.toLowerCase();
  if (r === "ok" || r === "success") return "성공";
  if (r === "error" || r === "fail") return "실패";
  return result;
}

function formatTs(ts: number | string): string {
  const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts);
  return d.toISOString().slice(0, 19).replace("T", " ");
}

function RbacPanel() {
  const rolesApi = useApi<RolesResponse>("/api/auth/roles");
  const tokensApi = useApi<IssuedToken[]>("/api/auth/tokens");

  const [subject, setSubject] = useState("");
  const [selectedRoles, setSelectedRoles] = useState<string[]>(["viewer"]);
  const [issued, setIssued] = useState<IssueTokenResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const availableRoles = rolesApi.data?.roles ?? [];

  const toggleRole = (role: string) =>
    setSelectedRoles((prev) =>
      prev.includes(role) ? prev.filter((r) => r !== role) : [...prev, role],
    );

  const issue = async () => {
    setBusy(true);
    setActionError(null);
    try {
      const res = await api<IssueTokenResponse>("/api/auth/tokens", {
        method: "POST",
        body: { subject, roles: selectedRoles },
      });
      setIssued(res);
      setCopied(false);
      setSubject("");
      tokensApi.reload();
    } catch (e) {
      setActionError(toMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const copySecret = async () => {
    if (!issued) return;
    try {
      await navigator.clipboard.writeText(issued.token);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const revoke = async (tokenId: string) => {
    setActionError(null);
    try {
      await api(`/api/auth/tokens/${tokenId}`, { method: "DELETE" });
      tokensApi.reload();
    } catch (e) {
      setActionError(toMessage(e));
    }
  };

  return (
    <Stack spacing={3}>
      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>역할 · 권한</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            역할 기반 접근통제(RBAC)에서 사용 가능한 역할 목록입니다. master key는 admin(<code>*</code>)으로 동작합니다.
          </Typography>
          {rolesApi.loading && !rolesApi.data ? (
            <Loading />
          ) : rolesApi.error ? (
            <ErrorView message={rolesApi.error} />
          ) : availableRoles.length === 0 ? (
            <EmptyView message="정의된 역할이 없습니다." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>역할</TableCell>
                  <TableCell>설명</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {availableRoles.map((role) => (
                  <TableRow key={role}>
                    <TableCell><Chip size="small" variant="outlined" label={role} /></TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {role === "admin" ? "전체 권한 (모든 동작)"
                          : role === "operator" ? "운영 동작 (학습·배포 등)"
                          : role === "approver" ? "릴리즈 승인/반려"
                          : role === "viewer" ? "읽기 전용 조회"
                          : "부여된 권한에 따름"}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>토큰 발급</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            역할을 부여한 API 토큰을 발급합니다. 발급된 토큰은 <code>Authorization: Bearer &lt;token&gt;</code> 으로 사용합니다.
          </Typography>
          <Stack spacing={2}>
            <TextField
              label="주체 (subject)"
              placeholder="사용자/서비스 식별자"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              fullWidth size="small"
            />
            <Box>
              <Typography variant="body2" fontWeight={600} sx={{ mb: 1 }}>역할 선택</Typography>
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1 }}>
                {availableRoles.map((role) => (
                  <Chip
                    key={role}
                    label={role}
                    size="small"
                    color={selectedRoles.includes(role) ? "primary" : "default"}
                    variant={selectedRoles.includes(role) ? "filled" : "outlined"}
                    onClick={() => toggleRole(role)}
                  />
                ))}
              </Stack>
            </Box>
            <Box>
              <Button
                variant="contained"
                onClick={issue}
                disabled={busy || !subject || selectedRoles.length === 0}
              >
                {busy ? "발급 중…" : "토큰 발급"}
              </Button>
            </Box>
          </Stack>

          {issued && (
            <Alert
              severity="success"
              sx={{ mt: 2 }}
              onClose={() => setIssued(null)}
              action={
                <Tooltip title={copied ? "복사됨" : "복사"}>
                  <IconButton size="small" onClick={copySecret} aria-label="토큰 복사">
                    <ContentCopyIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              }
            >
              <Typography variant="body2" fontWeight={600} gutterBottom>
                토큰이 발급되었습니다 (1회만 노출)
              </Typography>
              <TextField
                fullWidth size="small" value={issued.token}
                InputProps={{
                  readOnly: true,
                  sx: { fontFamily: "ui-monospace, monospace", fontSize: 13 },
                  startAdornment: (
                    <InputAdornment position="start">
                      <Chip size="small" variant="outlined" label={issued.subject} />
                    </InputAdornment>
                  ),
                }}
                onFocus={(e) => e.target.select()}
              />
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                {issued.note ?? "이 화면을 벗어나면 다시 확인할 수 없습니다."}
              </Typography>
            </Alert>
          )}
          {actionError && <ErrorView message={actionError} />}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>발급된 토큰</Typography>
          {tokensApi.loading && !tokensApi.data ? (
            <Loading />
          ) : tokensApi.error ? (
            <ErrorView message={tokensApi.error} />
          ) : !tokensApi.data || tokensApi.data.length === 0 ? (
            <EmptyView message="발급된 토큰이 없습니다." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>token_id</TableCell>
                  <TableCell>주체</TableCell>
                  <TableCell>역할</TableCell>
                  <TableCell align="right">액션</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {tokensApi.data.map((t) => (
                  <TableRow key={t.token_id}>
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                      {t.token_id}
                    </TableCell>
                    <TableCell>{t.subject}</TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", gap: 0.5 }}>
                        {t.roles.length === 0
                          ? <Typography variant="caption" color="text.secondary">—</Typography>
                          : t.roles.map((r) => <Chip key={r} size="small" variant="outlined" label={r} />)}
                      </Stack>
                    </TableCell>
                    <TableCell align="right">
                      <Button size="small" variant="outlined" color="error" onClick={() => revoke(t.token_id)}>
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
    </Stack>
  );
}

function AuditPanel() {
  const { data, loading, error } = useApi<AuditEntry[]>("/api/audit");

  return (
    <Card>
      <CardContent>
        <Typography variant="h3" gutterBottom>감사 로그</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          권한 동작(키·승인·배포·토큰)의 감사 추적을 시간 역순으로 표시합니다.
        </Typography>
        {loading && !data ? (
          <Loading />
        ) : error ? (
          <ErrorView message={error} />
        ) : !data || data.length === 0 ? (
          <EmptyView message="기록된 감사 로그가 없습니다." />
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>시각</TableCell>
                <TableCell>주체 (actor)</TableCell>
                <TableCell>동작 (action)</TableCell>
                <TableCell>대상 (resource)</TableCell>
                <TableCell>결과</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data.map((e, i) => (
                <TableRow key={`${e.actor}-${e.action}-${i}`}>
                  <TableCell sx={{ whiteSpace: "nowrap", fontSize: 12 }}>{formatTs(e.ts)}</TableCell>
                  <TableCell>{e.actor}</TableCell>
                  <TableCell><Typography variant="body2" fontWeight={600}>{e.action}</Typography></TableCell>
                  <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                    {e.target ?? "—"}
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      variant="outlined"
                      color={e.result === "ok" ? "success" : "error"}
                      label={resultLabel(e.result)}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

export default function Security() {
  const [tab, setTab] = useState(0);
  const [nonce, setNonce] = useState(0);
  const rolesT = useApi<RolesResponse>("/api/auth/roles", [nonce]);
  const tokensT = useApi<IssuedToken[]>("/api/auth/tokens", [nonce]);
  const auditT = useApi<AuditEntry[]>("/api/audit", [nonce]);
  const permsT = useApi<{ permissions: Record<string, string[]> }>("/api/auth/permissions", [nonce]);

  return (
    <>
      <PageHeader
        title="보안 · RBAC/감사"
        subtitle="역할 기반 접근통제 토큰 발급/폐기와 권한 동작 감사 추적"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => setNonce((n) => n + 1)} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      {(rolesT.data || tokensT.data || auditT.data) && (
        <SummaryTiles stats={[
          { label: "역할", value: rolesT.data?.roles?.length ?? 0, hint: "RBAC 정의", icon: SecurityOutlined },
          { label: "발급 토큰", value: tokensT.data?.length ?? 0, hint: "활성", icon: VpnKeyOutlined, accent: "success.main" },
          { label: "감사 로그", value: auditT.data?.length ?? 0, hint: "기록된 이벤트", icon: HistoryOutlined },
        ]} />
      )}

      <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 3 }}>
        <Tabs value={tab} onChange={(_, v: number) => setTab(v)}>
          <Tab label="RBAC" />
          <Tab label="감사 로그" />
        </Tabs>
      </Box>

      {tab === 0 && permsT.data && (() => {
        const perms = permsT.data.permissions;
        const roles = Object.keys(perms);
        const allPerms = Array.from(new Set(Object.values(perms).flat().filter((p) => p !== "*"))).sort();
        const has = (role: string, p: string) => perms[role]?.includes("*") || perms[role]?.includes(p);
        return (
          <Card variant="outlined" sx={{ borderRadius: 3, mb: 3 }}>
            <CardContent>
              <Typography variant="h3" gutterBottom>역할 × 권한 매트릭스</Typography>
              <Box sx={{ overflowX: "auto" }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>권한</TableCell>
                      {roles.map((r) => <TableCell key={r} align="center">{r}</TableCell>)}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {allPerms.map((p) => (
                      <TableRow key={p} hover>
                        <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{p}</TableCell>
                        {roles.map((r) => (
                          <TableCell key={r} align="center">
                            {has(r, p)
                              ? <CheckOutlined fontSize="small" sx={{ color: "success.main" }} />
                              : <Typography component="span" variant="caption" color="text.disabled">·</Typography>}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
                admin은 전체(<code>*</code>) 권한을 보유합니다.
              </Typography>
            </CardContent>
          </Card>
        );
      })()}

      {tab === 0 ? <RbacPanel key={`rbac-${nonce}`} /> : <AuditPanel key={`audit-${nonce}`} />}
    </>
  );
}

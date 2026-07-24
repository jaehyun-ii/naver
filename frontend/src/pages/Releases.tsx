import { useState } from "react";
import {
  Box, Button, Card, CardContent, Chip, Dialog, DialogActions, DialogContent,
  DialogContentText, DialogTitle, Stack, Table, TableBody, TableCell, TableHead,
  TableRow, TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import PendingActionsOutlined from "@mui/icons-material/PendingActionsOutlined";
import CheckCircleOutline from "@mui/icons-material/CheckCircleOutline";
import HighlightOffOutlined from "@mui/icons-material/HighlightOff";
import VerifiedOutlined from "@mui/icons-material/VerifiedOutlined";
import GavelOutlined from "@mui/icons-material/GavelOutlined";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView } from "../components/StateViews";
import { SummaryTiles, RichEmpty } from "../components/SummaryTiles";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

type ReleaseStatus = "Pending" | "Approved" | "Rejected";

interface ReleaseRequest {
  id: string;
  model_ref: string;
  suite?: string | null;
  metrics: Record<string, number>;
  auto_gate_passed: boolean;
  status: ReleaseStatus;
  requested_by?: string | null;
  approver?: string | null;
  reason?: string | null;
}

const STATUS_COLOR: Record<ReleaseStatus, "warning" | "success" | "error"> = {
  Pending: "warning",
  Approved: "success",
  Rejected: "error",
};

interface DecisionState {
  release: ReleaseRequest;
  kind: "approve" | "reject";
}

export default function Releases() {
  const { data, loading, error, reload } = useApi<ReleaseRequest[]>("/api/releases");
  const [decision, setDecision] = useState<DecisionState | null>(null);
  const [approver, setApprover] = useState("console-web");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const rels = data ?? [];
  const pending = rels.filter((r) => r.status === "Pending").length;
  const approved = rels.filter((r) => r.status === "Approved").length;
  const rejected = rels.filter((r) => r.status === "Rejected").length;
  const gatePassed = rels.filter((r) => r.auto_gate_passed).length;

  const openDialog = (release: ReleaseRequest, kind: "approve" | "reject") => {
    setDecision({ release, kind });
    setReason("");
    setActionError(null);
  };

  const submit = async () => {
    if (!decision) return;
    setBusy(true);
    setActionError(null);
    try {
      await api(`/api/releases/${decision.release.id}/${decision.kind}`, {
        method: "POST",
        body: { approver, reason: reason || undefined },
      });
      setDecision(null);
      reload();
    } catch (e) {
      setActionError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        title="승인 · 평가"
        subtitle="자동 게이트를 통과한 릴리즈의 승인/반려 거버넌스"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      {rels.length > 0 && (
        <SummaryTiles stats={[
          { label: "승인 대기", value: pending, hint: "결정 필요", icon: PendingActionsOutlined, accent: "warning.main" },
          { label: "승인됨", value: approved, icon: CheckCircleOutline, accent: "success.main" },
          { label: "반려됨", value: rejected, icon: HighlightOffOutlined, accent: "error.main" },
          { label: "게이트 통과", value: gatePassed, hint: `${rels.length}건 중`, icon: VerifiedOutlined },
        ]} />
      )}

      <Card variant="outlined" sx={{ borderRadius: 3 }}>
        <CardContent>
          {loading && !data ? (
            <Loading />
          ) : error ? (
            <ErrorView message={error} />
          ) : !data || data.length === 0 ? (
            <RichEmpty icon={GavelOutlined} title="승인 대기 중인 릴리즈가 없습니다"
              hint="파이프라인이 자동 게이트를 통과하면 후보 모델과 평가 지표가 승인 요청으로 여기에 올라옵니다."
              actionLabel="파이프라인 실행" to="/pipe" />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>모델</TableCell>
                  <TableCell>평가 지표</TableCell>
                  <TableCell>자동게이트</TableCell>
                  <TableCell>상태</TableCell>
                  <TableCell align="right">액션</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((r) => (
                  <TableRow key={r.id}>
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{r.model_ref}</TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                        {Object.entries(r.metrics).length === 0
                          ? <Typography variant="caption" color="text.secondary">—</Typography>
                          : Object.entries(r.metrics).map(([k, v]) => (
                              <Chip key={k} size="small" variant="outlined" label={`${k} ${v.toFixed(3)}`}
                                sx={{ fontFamily: "ui-monospace, monospace", fontSize: 11 }} />
                            ))}
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Chip
                        size="small" variant="outlined"
                        color={r.auto_gate_passed ? "success" : "default"}
                        label={r.auto_gate_passed ? "통과" : "미통과"}
                      />
                    </TableCell>
                    <TableCell><Chip size="small" color={STATUS_COLOR[r.status]} label={r.status} /></TableCell>
                    <TableCell align="right">
                      {r.status === "Pending" ? (
                        <Stack direction="row" spacing={1} justifyContent="flex-end">
                          <Button size="small" variant="contained" color="success" onClick={() => openDialog(r, "approve")}>
                            승인
                          </Button>
                          <Button size="small" variant="outlined" color="error" onClick={() => openDialog(r, "reject")}>
                            반려
                          </Button>
                        </Stack>
                      ) : (
                        <Typography variant="caption" color="text.secondary">
                          {r.approver ?? "—"}{r.reason ? ` · ${r.reason}` : ""}
                        </Typography>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Dialog open={!!decision} onClose={() => setDecision(null)} fullWidth maxWidth="xs">
        <DialogTitle>{decision?.kind === "approve" ? "릴리즈 승인" : "릴리즈 반려"}</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2, fontSize: 14 }}>
            모델 <code>{decision?.release.model_ref}</code>
          </DialogContentText>
          <Stack spacing={2}>
            <TextField label="승인자" value={approver} onChange={(e) => setApprover(e.target.value)} fullWidth size="small" />
            <TextField
              label={decision?.kind === "reject" ? "반려 사유" : "사유 (선택)"}
              value={reason} onChange={(e) => setReason(e.target.value)}
              fullWidth size="small" multiline minRows={2}
            />
          </Stack>
          {actionError && <Box sx={{ mt: 2 }}><ErrorView message={actionError} /></Box>}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDecision(null)}>취소</Button>
          <Button
            variant="contained"
            color={decision?.kind === "approve" ? "success" : "error"}
            onClick={submit}
            disabled={busy}
          >
            {busy ? "처리 중…" : decision?.kind === "approve" ? "승인" : "반려"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

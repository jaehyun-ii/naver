import { useState } from "react";
import {
  Box, Button, Card, CardContent, Chip, Dialog, DialogActions, DialogContent,
  DialogContentText, DialogTitle, Stack, Table, TableBody, TableCell, TableHead,
  TableRow, TextField, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
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

      <Card>
        <CardContent>
          {loading && !data ? (
            <Loading />
          ) : error ? (
            <ErrorView message={error} />
          ) : !data || data.length === 0 ? (
            <EmptyView message="승인 대기 중인 릴리즈가 없습니다." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>모델</TableCell>
                  <TableCell>Suite</TableCell>
                  <TableCell>자동게이트</TableCell>
                  <TableCell>상태</TableCell>
                  <TableCell align="right">액션</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((r) => (
                  <TableRow key={r.id}>
                    <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{r.model_ref}</TableCell>
                    <TableCell>{r.suite ?? "—"}</TableCell>
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

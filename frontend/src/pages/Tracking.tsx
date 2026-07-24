import { useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  Alert, Box, Button, Card, CardContent, Chip, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, Typography, IconButton, Tooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import CloudQueueOutlined from "@mui/icons-material/CloudQueueOutlined";
import ScienceOutlined from "@mui/icons-material/ScienceOutlined";
import Inventory2Outlined from "@mui/icons-material/Inventory2Outlined";
import LayersOutlined from "@mui/icons-material/LayersOutlined";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { SummaryTiles } from "../components/SummaryTiles";
import { useApi } from "../hooks/useApi";

interface Experiment {
  id: string;
  name: string;
  stage: string;
}
interface ExperimentsResp {
  available: boolean;
  error?: string;
  tracking_uri?: string;
  experiments?: Experiment[];
}

interface Run {
  run_id: string;
  status: string;
  start_time: number;
  metrics: Record<string, number>;
  params: Record<string, string>;
  tags: Record<string, string>;
}
interface RunsResp {
  available: boolean;
  error?: string;
  runs?: Run[];
}

interface ModelVersion {
  version: string;
  stage: string;
  run_id: string;
}
interface RegisteredModel {
  name: string;
  versions: ModelVersion[];
  aliases: Record<string, string>;
}
interface ModelsResp {
  available: boolean;
  error?: string;
  models?: RegisteredModel[];
}

function fmtTime(ms: number): string {
  if (!ms) return "—";
  try {
    return new Date(ms).toLocaleString("ko-KR");
  } catch {
    return String(ms);
  }
}

function runStatusColor(status: string): "success" | "error" | "warning" | "default" {
  const s = status.toUpperCase();
  if (s === "FINISHED") return "success";
  if (s === "FAILED" || s === "KILLED") return "error";
  if (s === "RUNNING" || s === "SCHEDULED") return "warning";
  return "default";
}

function RunsPanel({ experiment }: { experiment: Experiment }) {
  const { data, loading, error, reload } = useApi<RunsResp>(
    `/api/tracking/runs?experiment_id=${encodeURIComponent(experiment.id)}`,
  );
  const runs = data?.runs ?? [];

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
          <Typography variant="h3" sx={{ flexGrow: 1 }}>
            런 · {experiment.name}
          </Typography>
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="런 새로고침" size="small">
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {loading && !data ? (
          <Loading />
        ) : error ? (
          <ErrorView message={error} />
        ) : data && !data.available ? (
          <Alert severity="warning" sx={{ my: 2 }}>
            MLflow에 도달할 수 없습니다{data.error ? `: ${data.error}` : "."}
          </Alert>
        ) : runs.length === 0 ? (
          <EmptyView message="이 실험에는 아직 기록된 런이 없습니다." />
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>run</TableCell>
                <TableCell>상태</TableCell>
                <TableCell>시작 시각</TableCell>
                <TableCell>메트릭</TableCell>
                <TableCell>파라미터</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {runs.map((r) => (
                <TableRow key={r.run_id}>
                  <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                    {r.run_id.slice(0, 10)}
                  </TableCell>
                  <TableCell>
                    <Chip size="small" label={r.status} color={runStatusColor(r.status)} variant="outlined" />
                  </TableCell>
                  <TableCell sx={{ whiteSpace: "nowrap" }}>{fmtTime(r.start_time)}</TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", gap: 0.5 }}>
                      {Object.entries(r.metrics).length === 0 ? (
                        <Typography variant="caption" color="text.secondary">—</Typography>
                      ) : (
                        Object.entries(r.metrics).map(([k, v]) => (
                          <Chip key={k} size="small" variant="outlined" label={`${k}: ${Number(v).toFixed(4)}`} />
                        ))
                      )}
                    </Stack>
                  </TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", gap: 0.5 }}>
                      {Object.entries(r.params).length === 0 ? (
                        <Typography variant="caption" color="text.secondary">—</Typography>
                      ) : (
                        Object.entries(r.params).map(([k, v]) => (
                          <Chip key={k} size="small" variant="outlined" label={`${k}=${v}`} />
                        ))
                      )}
                    </Stack>
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

function ModelsPanel() {
  const { data, loading, error, reload } = useApi<ModelsResp>("/api/tracking/models");
  const models = data?.models ?? [];

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
          <Typography variant="h3" sx={{ flexGrow: 1 }}>모델 레지스트리</Typography>
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="모델 새로고침" size="small">
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {loading && !data ? (
          <Loading />
        ) : error ? (
          <ErrorView message={error} />
        ) : data && !data.available ? (
          <Alert severity="warning" sx={{ my: 2 }}>
            MLflow에 도달할 수 없습니다{data.error ? `: ${data.error}` : "."}
          </Alert>
        ) : models.length === 0 ? (
          <EmptyView message="등록된 모델이 없습니다." />
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>이름</TableCell>
                <TableCell>최신 버전</TableCell>
                <TableCell>별칭</TableCell>
                <TableCell align="right">액션</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {models.map((m) => (
                <TableRow key={m.name}>
                  <TableCell sx={{ fontWeight: 600 }}>{m.name}</TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", gap: 0.5 }}>
                      {m.versions.length === 0 ? (
                        <Typography variant="caption" color="text.secondary">—</Typography>
                      ) : (
                        m.versions.map((v) => (
                          <Chip
                            key={v.version}
                            size="small"
                            variant="outlined"
                            label={`v${v.version}${v.stage ? ` · ${v.stage}` : ""}`}
                          />
                        ))
                      )}
                    </Stack>
                  </TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", gap: 0.5 }}>
                      {Object.entries(m.aliases).length === 0 ? (
                        <Typography variant="caption" color="text.secondary">—</Typography>
                      ) : (
                        Object.entries(m.aliases).map(([alias, ver]) => (
                          <Chip key={alias} size="small" variant="outlined" label={`@${alias} → v${ver}`} />
                        ))
                      )}
                    </Stack>
                  </TableCell>
                  <TableCell align="right">
                    <Button size="small" component={RouterLink} to="/serving" startIcon={<CloudQueueOutlined />}>서빙</Button>
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

export default function Tracking() {
  const { data, loading, error, reload } = useApi<ExperimentsResp>("/api/tracking/experiments");
  const modelsApi = useApi<ModelsResp>("/api/tracking/models");
  const [selected, setSelected] = useState<Experiment | null>(null);

  const experiments = data?.experiments ?? [];
  const regModels = modelsApi.data?.models ?? [];
  const modelVersions = regModels.reduce((a, m) => a + m.versions.length, 0);

  return (
    <>
      <PageHeader
        title="실험·모델"
        subtitle="MLflow 실험·런·메트릭·모델 레지스트리 통합 뷰 (읽기전용)"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={() => reload()} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      {(data || modelsApi.data) && (
        <SummaryTiles stats={[
          { label: "실험", value: experiments.length, hint: "MLflow", icon: ScienceOutlined },
          { label: "등록 모델", value: regModels.length, hint: "레지스트리", icon: Inventory2Outlined, accent: "success.main" },
          { label: "모델 버전", value: modelVersions, hint: "누적", icon: LayersOutlined },
        ]} />
      )}

      {data?.tracking_uri && (
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 2 }}>
          tracking URI: <code>{data.tracking_uri}</code>
        </Typography>
      )}

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>실험 목록</Typography>
          {loading && !data ? (
            <Loading />
          ) : error ? (
            <ErrorView message={error} />
          ) : data && !data.available ? (
            <Alert severity="warning" sx={{ my: 2 }}>
              MLflow에 도달할 수 없습니다{data.error ? `: ${data.error}` : "."}
            </Alert>
          ) : experiments.length === 0 ? (
            <EmptyView message="등록된 실험이 없습니다." />
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>이름</TableCell>
                  <TableCell>ID</TableCell>
                  <TableCell>단계</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {experiments.map((e) => {
                  const active = selected?.id === e.id;
                  return (
                    <TableRow
                      key={e.id}
                      hover
                      selected={active}
                      onClick={() => setSelected(active ? null : e)}
                      sx={{ cursor: "pointer" }}
                    >
                      <TableCell sx={{ fontWeight: active ? 600 : 400 }}>{e.name}</TableCell>
                      <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>{e.id}</TableCell>
                      <TableCell>
                        <Chip size="small" variant="outlined" label={e.stage} />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
          {experiments.length > 0 && (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
              실험 행을 선택하면 해당 런 목록이 아래에 표시됩니다.
            </Typography>
          )}
        </CardContent>
      </Card>

      {selected && <RunsPanel experiment={selected} />}

      <ModelsPanel />
    </>
  );
}

import { useMemo, useState } from "react";
import {
  Card, CardContent, Chip, MenuItem, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, TextField, Typography, IconButton, Tooltip, Button,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import DownloadIcon from "@mui/icons-material/Download";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorView, EmptyView } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../api";

interface BucketInfo {
  name: string;
  created: string | null;
}
interface BucketsResponse {
  available: boolean;
  buckets?: BucketInfo[];
  error?: string;
}

interface ObjectInfo {
  key: string;
  size: number;
  modified: string | null;
}
interface ObjectsResponse {
  available: boolean;
  bucket?: string;
  objects?: ObjectInfo[];
  truncated?: boolean;
  error?: string;
}

interface PresignResponse {
  available: boolean;
  url?: string;
  error?: string;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(1)} ${units[i]}`;
}

function toApiMessage(e: unknown): string {
  return e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : (e as Error).message;
}

export default function Storage() {
  const buckets = useApi<BucketsResponse>("/api/storage/buckets");

  const [bucket, setBucket] = useState<string>("");
  const [prefix, setPrefix] = useState<string>("");
  const [actionError, setActionError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);

  const objectsPath = useMemo(
    () => `/api/storage/objects?bucket=${encodeURIComponent(bucket)}&prefix=${encodeURIComponent(prefix)}`,
    [bucket, prefix],
  );
  const objects = useApi<ObjectsResponse>(objectsPath, [bucket, prefix]);

  const bucketList = buckets.data?.available ? buckets.data.buckets ?? [] : [];
  const objectList = objects.data?.available ? objects.data.objects ?? [] : [];

  const reloadAll = () => {
    buckets.reload();
    if (bucket) objects.reload();
  };

  const openPresign = async (key: string) => {
    setActionError(null);
    setBusyKey(key);
    try {
      const r = await api<PresignResponse>(
        `/api/storage/presign?bucket=${encodeURIComponent(bucket)}&key=${encodeURIComponent(key)}`,
      );
      if (!r.available || !r.url) throw new Error(r.error ?? "presigned URL 발급 실패");
      window.open(r.url, "_blank", "noopener,noreferrer");
    } catch (e) {
      setActionError(toApiMessage(e));
    } finally {
      setBusyKey(null);
    }
  };

  const downloadProxy = (key: string) => {
    const url = `/api/storage/download?bucket=${encodeURIComponent(bucket)}&key=${encodeURIComponent(key)}`;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  return (
    <>
      <PageHeader
        title="스토리지"
        subtitle="MinIO/S3 버킷·객체 브라우저 (읽기전용)"
        action={
          <Tooltip title="새로고침">
            <IconButton onClick={reloadAll} aria-label="새로고침"><RefreshIcon /></IconButton>
          </Tooltip>
        }
      />

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h3" gutterBottom>버킷 선택</Typography>
          {buckets.loading && !buckets.data ? (
            <Loading />
          ) : buckets.error ? (
            <ErrorView message={buckets.error} />
          ) : buckets.data && !buckets.data.available ? (
            <ErrorView message={`스토리지에 연결할 수 없습니다: ${buckets.data.error ?? "알 수 없는 오류"}`} />
          ) : bucketList.length === 0 ? (
            <EmptyView message="사용 가능한 버킷이 없습니다." />
          ) : (
            <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap", gap: 2 }}>
              <TextField
                select
                label="버킷"
                value={bucket}
                onChange={(e) => { setBucket(e.target.value); setPrefix(""); setActionError(null); }}
                sx={{ minWidth: 240 }}
                size="small"
              >
                {bucketList.map((b) => (
                  <MenuItem key={b.name} value={b.name}>{b.name}</MenuItem>
                ))}
              </TextField>
              <TextField
                label="접두사 (prefix)"
                value={prefix}
                onChange={(e) => setPrefix(e.target.value)}
                placeholder="예: datasets/"
                sx={{ minWidth: 280 }}
                size="small"
                disabled={!bucket}
              />
            </Stack>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h3" gutterBottom>객체 목록</Typography>

          {actionError && <ErrorView message={actionError} />}

          {!bucket ? (
            <EmptyView message="위에서 버킷을 선택하세요." />
          ) : objects.loading && !objects.data ? (
            <Loading />
          ) : objects.error ? (
            <ErrorView message={objects.error} />
          ) : objects.data && !objects.data.available ? (
            <ErrorView message={`객체를 불러올 수 없습니다: ${objects.data.error ?? "알 수 없는 오류"}`} />
          ) : objectList.length === 0 ? (
            <EmptyView message="해당 접두사에 객체가 없습니다." />
          ) : (
            <>
              {objects.data?.truncated && (
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                  결과가 잘렸습니다 (최대 200건). 접두사로 범위를 좁혀 보세요.
                </Typography>
              )}
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>키</TableCell>
                    <TableCell align="right">크기</TableCell>
                    <TableCell align="right">수정 시각</TableCell>
                    <TableCell align="right">작업</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {objectList.map((o) => (
                    <TableRow key={o.key} hover>
                      <TableCell sx={{ fontFamily: "ui-monospace, monospace", fontSize: 12, wordBreak: "break-all" }}>
                        {o.key}
                      </TableCell>
                      <TableCell align="right">
                        <Chip size="small" variant="outlined" label={formatSize(o.size)} />
                      </TableCell>
                      <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                        {o.modified ? new Date(o.modified).toLocaleString() : "—"}
                      </TableCell>
                      <TableCell align="right">
                        <Stack direction="row" spacing={1} justifyContent="flex-end">
                          <Tooltip title="presigned 링크 열기">
                            <span>
                              <Button
                                size="small"
                                variant="outlined"
                                startIcon={<OpenInNewIcon />}
                                onClick={() => openPresign(o.key)}
                                disabled={busyKey === o.key}
                              >
                                링크
                              </Button>
                            </span>
                          </Tooltip>
                          <Tooltip title="콘솔 프록시로 다운로드">
                            <Button
                              size="small"
                              variant="outlined"
                              startIcon={<DownloadIcon />}
                              onClick={() => downloadProxy(o.key)}
                            >
                              다운로드
                            </Button>
                          </Tooltip>
                        </Stack>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </>
          )}
        </CardContent>
      </Card>
    </>
  );
}

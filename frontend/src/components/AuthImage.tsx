import { useEffect, useState } from "react";
import { Box, Skeleton, Typography } from "@mui/material";
import { apiBlob } from "../api";

// 인증 헤더(X-Master-Key)가 필요한 이미지 — <img src>로는 헤더를 실을 수 없어
// apiBlob으로 받아 blob URL을 만들어 표시한다(언마운트 시 revoke).
export function AuthImage({ src, alt, maxHeight = 300 }: { src: string; alt?: string; maxHeight?: number }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    let obj: string | null = null;
    setUrl(null); setFailed(false);
    apiBlob(src)
      .then((b) => { obj = URL.createObjectURL(b); if (alive) setUrl(obj); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; if (obj) URL.revokeObjectURL(obj); };
  }, [src]);

  if (failed) {
    return <Typography variant="caption" color="text.secondary">🖼 이미지를 불러올 수 없음{alt ? ` — ${alt}` : ""}</Typography>;
  }
  if (!url) return <Skeleton variant="rectangular" height={120} sx={{ borderRadius: 1, maxWidth: 360 }} />;
  return (
    <Box component="img" src={url} alt={alt ?? ""} loading="lazy"
      sx={{ maxWidth: "100%", maxHeight, display: "block", borderRadius: 1, border: 1, borderColor: "divider", bgcolor: "background.paper" }} />
  );
}

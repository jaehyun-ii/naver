// 백엔드 콘솔 API 클라이언트.
// 인증: X-Master-Key 헤더 (기존 콘솔과 동일). 키는 localStorage에 보관한다.

const MASTER_KEY_STORAGE = "llmops.masterKey";

export function getMasterKey(): string {
  return localStorage.getItem(MASTER_KEY_STORAGE) ?? "";
}

export function setMasterKey(key: string): void {
  if (key) localStorage.setItem(MASTER_KEY_STORAGE, key);
  else localStorage.removeItem(MASTER_KEY_STORAGE);
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface ApiOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
}

export async function api<T = unknown>(path: string, opts: ApiOptions = {}): Promise<T> {
  const { method = "GET", body, signal } = opts;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const master = getMasterKey();
  if (master) headers["X-Master-Key"] = master;

  const res = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = (data as { detail?: string }).detail || res.statusText;
    throw new ApiError(detail, res.status);
  }
  return data as T;
}

// 멀티파트 업로드(파일). Content-Type은 브라우저가 boundary와 함께 설정하도록 비운다.
export async function apiUpload<T = unknown>(path: string, form: FormData): Promise<T> {
  const headers: Record<string, string> = {};
  const master = getMasterKey();
  if (master) headers["X-Master-Key"] = master;
  const res = await fetch(path, { method: "POST", headers, body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = (data as { detail?: string }).detail || res.statusText;
    throw new ApiError(detail, res.status);
  }
  return data as T;
}

// 인증 헤더를 실어 바이너리(PDF 등)를 blob으로 받는다(인라인 뷰어용).
export async function apiBlob(path: string): Promise<Blob> {
  const headers: Record<string, string> = {};
  const master = getMasterKey();
  if (master) headers["X-Master-Key"] = master;
  const res = await fetch(path, { headers });
  if (!res.ok) throw new ApiError(res.statusText, res.status);
  return res.blob();
}

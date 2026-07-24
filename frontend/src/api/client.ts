import axios from "axios";

// Dev (npm run dev on :5173): talk to the local backend directly.
// Production build: same-origin relative paths — the FastAPI server (or a
// reverse proxy / Cloudflare Tunnel in front of it) serves both the SPA
// and /api, so no CORS is involved.
const API_BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

const api = axios.create({ baseURL: API_BASE });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      // Chỉ ép reload về /login khi PHIÊN thực sự hết hạn (đang có token).
      // Nếu 401 mà không hề có token (request lỡ bắn lúc chưa đăng nhập) thì
      // đừng reload — nếu không sẽ lặp vô hạn ngay trên trang đăng nhập.
      const hadToken = !!localStorage.getItem("token");
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      if (hadToken) window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);

export default api;

// ── Auth ──────────────────────────────────────────────────
export const authAPI = {
  register: (email: string, password: string, name: string) =>
    api.post("/api/auth/register", { email, password, name }).then((r) => r.data),
  login: (email: string, password: string) =>
    api.post("/api/auth/login", { email, password }).then((r) => r.data),
  /** Refresh the signed-in user's server-side attributes (e.g. is_admin). */
  me: (): Promise<{ user_id: string; email: string; is_admin: boolean }> =>
    api.get("/api/auth/me").then((r) => r.data),
};

/** Turn an optional 0–100 progress callback into an axios upload-progress
 *  handler; undefined when no callback is given so axios skips the work. */
const uploadProgress = (onProgress?: (pct: number) => void) =>
  onProgress
    ? (e: { loaded: number; total?: number }) =>
        onProgress(e.total ? Math.round((e.loaded / e.total) * 100) : 0)
    : undefined;

// ── Sessions ──────────────────────────────────────────────
export const sessionsAPI = {
  list: () => api.get("/api/sessions").then((r) => r.data),
  create: () => api.post("/api/sessions").then((r) => r.data),
  delete: (id: string) => api.delete(`/api/sessions/${id}`).then((r) => r.data),
  messages: (id: string) => api.get(`/api/sessions/${id}/messages`).then((r) => r.data),
  feedback: (messageId: string, rating: "up" | "down" | null) =>
    api.post(`/api/messages/${messageId}/feedback`, { rating }).then((r) => r.data),
};

// ── Streaming chat (SSE over fetch — axios can't stream) ──
export interface StreamMeta { sources: object[]; complexity: string; n_hops?: number }
export interface MultihopStep {
  stage: "hop_start" | "hop_done";
  hop: number;
  total?: number;
  query?: string;
  answer?: string;
  titles?: string[];
}
export interface StreamDone {
  latency_ms: number;
  message_id?: string;
  user_message_id?: string;
  session_title?: string;
}
export interface GroundingResult { total: number; verified: number; unsupported: string[] }

export interface StreamCallbacks {
  onMeta?: (meta: StreamMeta) => void;
  onStep?: (step: MultihopStep) => void;
  onDelta?: (text: string) => void;
  onDone?: (done: StreamDone) => void;
  /** Follow-up question chips — arrives AFTER done (server generates them post-answer) */
  onSuggestions?: (questions: string[]) => void;
  /** Grounding-check result — arrives AFTER done (cited-claim verification) */
  onGrounding?: (grounding: GroundingResult) => void;
  onError?: (detail: string) => void;
}

export async function chatStream(
  sessionId: string,
  body: {
    question: string;
    history: object[];
    attachments?: string[];
    regenerate_message_id?: string;
    include_doc_ids?: string[];
    use_global_kb?: boolean;
  },
  cb: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const token = localStorage.getItem("token");
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
    signal,
  });
  if (res.status === 401) {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    window.location.href = "/login";
    return;
  }
  if (!res.ok || !res.body) {
    // Empty when the error body has no server detail — the caller localizes
    // the fallback message (see useChatStream's onError).
    let detail = "";
    try {
      detail = (await res.json()).detail || detail;
    } catch { /* non-JSON error body */ }
    cb.onError?.(detail);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let sep;
    // SSE frames end with a blank line
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      try {
        const parsed = JSON.parse(data);
        if (event === "meta") cb.onMeta?.(parsed);
        else if (event === "step") cb.onStep?.(parsed);
        else if (event === "delta") cb.onDelta?.(parsed.text);
        else if (event === "done") cb.onDone?.(parsed);
        else if (event === "suggestions") cb.onSuggestions?.(parsed.questions ?? []);
        else if (event === "grounding") cb.onGrounding?.(parsed);
        else if (event === "error") cb.onError?.(parsed.detail);
      } catch { /* skip malformed frame */ }
    }
  }
}

// ── Documents ─────────────────────────────────────────────
export interface DocContent {
  id: string;
  name: string;
  type: string;
  has_file: boolean;
  text: string;
  chunks: { id: string; start: number; end: number; page?: number }[];
}

/** Auth headers for direct fetch/react-pdf requests that bypass axios. */
export const authHeaders = (): Record<string, string> => {
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
};

export const documentsAPI = {
  list: (sessionId?: string) =>
    api.get("/api/documents", { params: sessionId ? { session_id: sessionId } : {} })
      .then((r) => r.data),
  content: (docId: string): Promise<DocContent> =>
    api.get(`/api/documents/${docId}/content`).then((r) => r.data),
  fileUrl: (docId: string) => `${API_BASE}/api/documents/${docId}/file`,
  /** Original file bytes as a Blob (auth header applied) — for image thumbnails. */
  fileBlob: (docId: string): Promise<Blob> =>
    api.get(`/api/documents/${docId}/file`, { responseType: "blob" }).then((r) => r.data),
  uploadFile: (file: File, sessionId: string, onProgress?: (pct: number) => void) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("session_id", sessionId);
    return api.post("/api/documents/upload", fd, { onUploadProgress: uploadProgress(onProgress) })
      .then((r) => r.data);
  },
  uploadUrl: (url: string, sessionId: string) => {
    const fd = new FormData();
    fd.append("url", url);
    fd.append("session_id", sessionId);
    return api.post("/api/documents/upload", fd).then((r) => r.data);
  },
  /** Speech-to-text for voice input (fallback when the browser lacks the Web Speech API). */
  transcribe: (file: File): Promise<{ text: string }> => {
    const fd = new FormData();
    fd.append("file", file);
    return api.post("/api/documents/transcribe", fd).then((r) => r.data);
  },
  delete: (id: string) => api.delete(`/api/documents/${id}`).then((r) => r.data),
};

// ── Global knowledge base ─────────────────────────────────
export interface KbDoc {
  id: string;
  name: string;
  type: string;
  chunk_count: number;
  created_at: string;
}

export interface KbChunk {
  id: string;
  text: string;
  chunk_index: number;
  page?: number;
}

export const knowledgeAPI = {
  list: (): Promise<{ documents: KbDoc[]; total_chunks: number; can_manage: boolean }> =>
    api.get("/api/knowledge").then((r) => r.data),
  content: (docId: string): Promise<DocContent> =>
    api.get(`/api/knowledge/${docId}/content`).then((r) => r.data),
  fileUrl: (docId: string) => `${API_BASE}/api/knowledge/${docId}/file`,
  chunks: (docId: string): Promise<{ doc_id: string; name: string; chunks: KbChunk[] }> =>
    api.get(`/api/knowledge/${docId}/chunks`).then((r) => r.data),
  updateChunk: (docId: string, chunkId: string, text: string) =>
    api.patch(`/api/knowledge/${docId}/chunks/${chunkId}`, { text }).then((r) => r.data),
  deleteChunk: (docId: string, chunkId: string) =>
    api.delete(`/api/knowledge/${docId}/chunks/${chunkId}`).then((r) => r.data),
  uploadFile: (file: File, onProgress?: (pct: number) => void) => {
    const fd = new FormData();
    fd.append("file", file);
    return api.post("/api/knowledge/upload", fd, { onUploadProgress: uploadProgress(onProgress) })
      .then((r) => r.data);
  },
  uploadUrl: (url: string) => {
    const fd = new FormData();
    fd.append("url", url);
    return api.post("/api/knowledge/upload", fd).then((r) => r.data);
  },
  delete: (docId: string) => api.delete(`/api/knowledge/${docId}`).then((r) => r.data),
  reload: () => api.post("/api/knowledge/reload").then((r) => r.data),
};

// ── Monitoring ────────────────────────────────────────────
export interface MonitoringStats {
  window_days: number;
  n_queries: number;
  errors: number;
  aborted: number;
  error_rate: number;
  latency_ms: { avg: number; p50: number; p95: number; p99: number };
  total_cost_usd: number;
  total_tokens: number;
  by_complexity: Record<string, number>;
  multihop_queries: number;
  feedback: { up: number; down: number };
  per_day: { date: string; queries: number; cost_usd: number; multihop: number }[];
}

export interface DownvotedItem {
  message_id: string;
  session_id: string;
  question: string;
  answer: string;
  sources: { rank: number; title: string; snippet: string }[];
  created_at: string;
  in_eval_queue: boolean;
}

export const monitoringAPI = {
  stats: (days = 14): Promise<MonitoringStats> =>
    api.get("/api/monitoring/stats", { params: { days } }).then((r) => r.data),
  downvoted: (days = 30): Promise<{ items: DownvotedItem[] }> =>
    api.get("/api/monitoring/downvoted", { params: { days } }).then((r) => r.data),
  addToEvalQueue: (messageId: string, groundTruth?: string): Promise<{ success: boolean; already_queued: boolean; total: number }> =>
    api.post("/api/monitoring/eval-queue", { message_id: messageId, ground_truth: groundTruth })
      .then((r) => r.data),
};

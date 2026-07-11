import axios from "axios";

const api = axios.create({ baseURL: "http://localhost:8000" });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      window.location.href = "/login";
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
};

// ── Sessions ──────────────────────────────────────────────
export const sessionsAPI = {
  list: () => api.get("/api/sessions").then((r) => r.data),
  create: () => api.post("/api/sessions").then((r) => r.data),
  delete: (id: string) => api.delete(`/api/sessions/${id}`).then((r) => r.data),
  messages: (id: string) => api.get(`/api/sessions/${id}/messages`).then((r) => r.data),
  chat: (id: string, question: string, history: object[], attachments: string[] = []) =>
    api.post(`/api/sessions/${id}/chat`, { question, history, attachments }).then((r) => r.data),
};

// ── Documents ─────────────────────────────────────────────
export const documentsAPI = {
  list: () => api.get("/api/documents").then((r) => r.data),
  uploadFile: (file: File, sessionId: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("session_id", sessionId);
    return api.post("/api/documents/upload", fd).then((r) => r.data);
  },
  uploadUrl: (url: string, sessionId: string) => {
    const fd = new FormData();
    fd.append("url", url);
    fd.append("session_id", sessionId);
    return api.post("/api/documents/upload", fd).then((r) => r.data);
  },
  delete: (id: string) => api.delete(`/api/documents/${id}`).then((r) => r.data),
};

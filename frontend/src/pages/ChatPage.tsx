import { useState, useEffect, useRef, useCallback } from "react";
import {
  Send, Mic, Paperclip, X, Globe,
  Loader2, Square, CheckCircle2, AlertCircle,
  BookOpen, FileText, SlidersHorizontal,
} from "lucide-react";
import { sessionsAPI, documentsAPI, knowledgeAPI, chatStream } from "../api/client";
import MessageBubble from "../components/MessageBubble";
import AiRobotIcon from "../components/AiRobotIcon";
import DocViewerPanel from "../components/DocViewerPanel";
import type { ViewerTarget } from "../components/DocViewerPanel";
import type { Message, Source } from "../components/MessageBubble";

interface Props { sessionId: string | null; onSessionCreated: (id: string) => void; }

interface Attachment {
  id: string;
  name: string;
  status: "uploading" | "done" | "error";
  detail?: string;
}

interface SessionDoc {
  id: string;
  name: string;
  type: string;
  chunk_count: number;
}

const DOC_ACCEPT = ".pdf,.txt";
const IMAGE_ACCEPT = ".png,.jpg,.jpeg,.webp";
const AUDIO_ACCEPT = ".mp3,.wav,.m4a,.ogg,.webm";
const FILE_ACCEPT = `${DOC_ACCEPT},${IMAGE_ACCEPT},${AUDIO_ACCEPT}`;

const URL_REGEX = /^https?:\/\/\S+$/i;
const MAX_INPUT_HEIGHT = 120;

export default function ChatPage({ sessionId, onSessionCreated }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [urlTip, setUrlTip] = useState(false);
  const [recording, setRecording] = useState(false);
  const [inputFocused, setInputFocused] = useState(false);
  const [viewerTarget, setViewerTarget] = useState<ViewerTarget | null>(null);
  // Source picker: which session docs + the shared KB take part in answers
  const [sessionDocs, setSessionDocs] = useState<SessionDoc[]>([]);
  const [kbDocCount, setKbDocCount] = useState(0);
  const [excludedDocs, setExcludedDocs] = useState<Set<string>>(new Set());
  const [useGlobalKb, setUseGlobalKb] = useState(true);
  const [pickerOpen, setPickerOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const sessionIdRef = useRef<string | null>(sessionId);
  const pendingDocsRef = useRef<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (sessionIdRef.current !== sessionId) {
      pendingDocsRef.current = [];
      prevCountRef.current = 0; // reset so bulk-load scroll fires correctly
      setAttachments([]);
      setInputFocused(false);
      setViewerTarget(null);
      setExcludedDocs(new Set());
      setUseGlobalKb(true);
      setPickerOpen(false);
    }
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  const refreshSessionDocs = useCallback((sid: string | null) => {
    if (!sid) { setSessionDocs([]); return; }
    documentsAPI.list(sid)
      .then((data) => setSessionDocs(data.documents || []))
      .catch(() => setSessionDocs([]));
  }, []);

  useEffect(() => { refreshSessionDocs(sessionId); }, [sessionId, refreshSessionDocs]);

  useEffect(() => {
    knowledgeAPI.list()
      .then((data) => setKbDocCount((data.documents || []).length))
      .catch(() => setKbDocCount(0));
  }, []);

  const ensureSession = useCallback(async (notify: boolean) => {
    if (sessionIdRef.current) return sessionIdRef.current;
    const s = await sessionsAPI.create();
    sessionIdRef.current = s.session_id;
    if (notify) onSessionCreated(s.session_id);
    return s.session_id;
  }, [onSessionCreated]);

  useEffect(() => {
    if (!sessionId) { setMessages([]); return; }
    let cancelled = false;
    let timer: number | undefined;
    const load = (attempt = 0) => {
      sessionsAPI.messages(sessionId)
        .then((data) => { if (!cancelled) setMessages(data); })
        .catch(() => {
          if (!cancelled && attempt < 5) {
            timer = window.setTimeout(() => load(attempt + 1), 2000);
          }
        });
    };
    load();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [sessionId]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const newCount = messages.length;
    const prevCount = prevCountRef.current;
    prevCountRef.current = newCount;

    if (newCount === 0) return; // session cleared

    const isStreaming = messages.some((m) => m.streaming);

    // Bulk load (switched session): wait one paint cycle then jump to bottom
    if (prevCount === 0 && newCount > 1) {
      requestAnimationFrame(() => {
        el.scrollTo({ top: el.scrollHeight, behavior: "auto" });
      });
      return;
    }
    // New single message appended (user sent or AI placeholder added): smooth scroll
    if (newCount === prevCount + 1) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
      return;
    }
    // Streaming in progress: keep following new tokens
    if (isStreaming) {
      el.scrollTo({ top: el.scrollHeight, behavior: "auto" });
    }
    // Source expand, rating, feedback → do NOT hijack scroll position
  }, [messages]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_INPUT_HEIGHT)}px`;
    el.style.overflowY = el.scrollHeight > MAX_INPUT_HEIGHT ? "auto" : "hidden";
  }, [input]);

  const uploadFiles = useCallback(async (files: File[]) => {
    let sid: string;
    try {
      sid = await ensureSession(true);
    } catch {
      setAttachments((p) => [...p, {
        id: `${Date.now()}-session`, name: files[0]?.name || "Tệp",
        status: "error", detail: "Không tạo được phiên chat",
      }]);
      return;
    }
    for (const file of files) {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      setAttachments((p) => [...p, { id, name: file.name, status: "uploading" }]);
      try {
        const res = await documentsAPI.uploadFile(file, sid);
        pendingDocsRef.current.push(file.name);
        setAttachments((p) => p.map((a) =>
          a.id === id ? { ...a, status: "done" as const, detail: `${res.chunk_count} chunks` } : a
        ));
      } catch (e: any) {
        setAttachments((p) => p.map((a) =>
          a.id === id ? { ...a, status: "error" as const, detail: e?.response?.data?.detail || "Lỗi upload" } : a
        ));
      }
    }
    refreshSessionDocs(sid);
  }, [ensureSession, refreshSessionDocs]);

  const addUrlAsDocument = useCallback(async (url: string) => {
    const id = `${Date.now()}-url`;
    setAttachments((p) => [...p, { id, name: url, status: "uploading" }]);
    try {
      const sid = await ensureSession(true);
      const res = await documentsAPI.uploadUrl(url, sid);
      pendingDocsRef.current.push(res.name || url);
      setAttachments((p) => p.map((a) =>
        a.id === id ? { ...a, name: res.name || url, status: "done" as const, detail: `${res.chunk_count} chunks` } : a
      ));
      refreshSessionDocs(sid);
    } catch (e: any) {
      setAttachments((p) => p.map((a) =>
        a.id === id ? { ...a, status: "error" as const, detail: e?.response?.data?.detail || "Không thể fetch URL" } : a
      ));
    }
  }, [ensureSession, refreshSessionDocs]);

  const openPicker = useCallback((accept: string) => {
    const el = fileInputRef.current;
    if (!el) return;
    el.accept = accept;
    el.click();
  }, []);

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const files = Array.from(e.clipboardData.files);
    if (files.length > 0) {
      e.preventDefault();
      uploadFiles(files);
      return;
    }
    const text = e.clipboardData.getData("text").trim();
    if (URL_REGEX.test(text)) {
      e.preventDefault();
      setUrlTip(false);
      addUrlAsDocument(text);
    }
  }, [uploadFiles, addUrlAsDocument]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const parts: Blob[] = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) parts.push(e.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const mime = recorder.mimeType || "audio/webm";
        const ext = mime.includes("ogg") ? "ogg" : "webm";
        const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
        const file = new File(parts, `ghi-am-${stamp}.${ext}`, { type: mime });
        uploadFiles([file]);
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch {
      setAttachments((p) => [...p, {
        id: `${Date.now()}-mic`, name: "Micro", status: "error",
        detail: "Không truy cập được micro",
      }]);
    }
  }, [uploadFiles]);

  const stopRecording = useCallback(() => {
    recorderRef.current?.stop();
    recorderRef.current = null;
    setRecording(false);
  }, []);

  const patchMessage = useCallback((id: string, patch: Partial<Message>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  /** Drive one SSE exchange into the placeholder assistant message `aiId`. */
  const runStream = useCallback(async (
    sid: string,
    aiId: string,
    body: { question: string; history: object[]; attachments?: string[]; regenerate_message_id?: string },
  ) => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let content = "";
    try {
      await chatStream(sid, body, {
        onMeta: (meta) => patchMessage(aiId, { sources: meta.sources as Source[] }),
        onDelta: (text) => {
          content += text;
          patchMessage(aiId, { content });
        },
        onDone: (done) => {
          // Swap in the DB id last so feedback targets the persisted row.
          patchMessage(aiId, { streaming: false, ...(done.message_id ? { id: done.message_id } : {}) });
        },
        onError: (detail) => {
          content = content || detail;
          patchMessage(aiId, { content, streaming: false });
        },
      }, ctrl.signal);
    } catch (err: any) {
      if (err?.name !== "AbortError") {
        patchMessage(aiId, {
          content: content || "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.",
          streaming: false,
        });
      }
      // Aborted: keep the partial answer (backend persists it too)
    } finally {
      abortRef.current = null;
      setMessages((prev) => prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)));
      setSending(false);
    }
  }, [patchMessage]);

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  /** Chat-body fields reflecting the source picker state. */
  const sourceParams = useCallback(() => ({
    ...(excludedDocs.size > 0
      ? { include_doc_ids: sessionDocs.filter((d) => !excludedDocs.has(d.id)).map((d) => d.id) }
      : {}),
    use_global_kb: useGlobalKb,
  }), [excludedDocs, sessionDocs, useGlobalKb]);

  const openSource = useCallback((s: Source) => {
    if (!s.doc_id) return;
    setViewerTarget({
      docId: s.doc_id,
      scope: s.scope === "global" ? "global" : "session",
      title: s.title,
      page: s.page,
      chunkId: s.chunk_id,
      snippet: s.snippet,
    });
  }, []);

  const send = useCallback(async () => {
    const q = input.trim();
    if (!q || sending) return;

    const isNewSession = !sessionIdRef.current;
    const sid = await ensureSession(false);

    const attached = pendingDocsRef.current;
    pendingDocsRef.current = [];
    setAttachments((p) => p.filter((a) => a.status !== "done"));

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: q, attachments: attached };
    const aiId = (Date.now() + 1).toString();
    setMessages((prev) => [...prev, userMsg, { id: aiId, role: "assistant", content: "", streaming: true }]);
    setInput("");
    setSending(true);

    const history = messages.slice(-6).map((m) => ({ role: m.role, content: m.content }));
    await runStream(sid, aiId, { question: q, history, attachments: attached, ...sourceParams() });
    if (isNewSession) onSessionCreated(sid);
  }, [input, sending, messages, ensureSession, onSessionCreated, runStream, sourceParams]);

  const regenerate = useCallback(async () => {
    if (sending || !sessionIdRef.current) return;
    // Last assistant message + the user question that produced it
    let ai = messages.length - 1;
    while (ai >= 0 && messages[ai].role !== "assistant") ai--;
    if (ai < 0) return;
    let ui = ai - 1;
    while (ui >= 0 && messages[ui].role !== "user") ui--;
    if (ui < 0) return;

    const aiMsg = messages[ai];
    const question = messages[ui].content;
    const history = messages.slice(0, ui).slice(-6).map((m) => ({ role: m.role, content: m.content }));

    patchMessage(aiMsg.id, { content: "", sources: undefined, feedback: null, streaming: true });
    setSending(true);
    await runStream(sessionIdRef.current, aiMsg.id, {
      question,
      history,
      // Only persisted messages (uuid ids) can be replaced server-side
      ...(aiMsg.id.includes("-") ? { regenerate_message_id: aiMsg.id } : {}),
      ...sourceParams(),
    });
  }, [messages, sending, patchMessage, runStream, sourceParams]);

  const handleFeedback = useCallback(async (messageId: string, rating: "up" | "down" | null) => {
    patchMessage(messageId, { feedback: rating });
    try {
      await sessionsAPI.feedback(messageId, rating);
    } catch {
      patchMessage(messageId, { feedback: null });
    }
  }, [patchMessage]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const isEmpty = messages.length === 0;


  return (
    <div
      className="flex flex-1 h-full overflow-hidden"
      style={{ 
        background: "rgba(255,255,255,0.3)", 
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        boxShadow: "-10px 0 30px rgba(0,0,0,0.02)"
      }}
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files || []);
          if (files.length) uploadFiles(files);
          e.target.value = "";
        }}
      />

      {/* Main chat */}
      <div className="flex flex-col flex-1 min-w-0">


        {/* Messages */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto"
          style={{ overflowAnchor: "none" }}
        >
          {isEmpty ? (
            /* ── Greeting: fades & slides up when input is focused ── */
            <div
              className="h-full flex flex-col items-center justify-center text-center"
              style={{
                transition: 'opacity 0.5s ease, transform 0.55s cubic-bezier(0.4,0,0.2,1)',
                opacity: inputFocused ? 0 : 1,
                transform: inputFocused ? 'translateY(-60px) scale(0.94)' : 'translateY(0) scale(1)',
                pointerEvents: inputFocused ? 'none' : 'auto',
              }}
            >
              <AiRobotIcon size={150} />
              <h2 className="text-2xl font-bold text-[#1A1A2E] mt-0 mb-2">
                Xin chào! 👋
              </h2>
              <p className="text-sm" style={{ color: "#6B7280" }}>
                <strong style={{ color: "#7C3AED" }}>How can I help you?</strong>
              </p>
            </div>
          ) : (
            /* ── Messages ── */
            <div className="px-20 pt-6 pb-16">
              {messages.map((m, i) => (
                <MessageBubble
                  key={m.id}
                  msg={m}
                  onFeedback={handleFeedback}
                  onOpenSource={openSource}
                  onRegenerate={
                    !sending && m.role === "assistant" && i === messages.length - 1
                      ? regenerate
                      : undefined
                  }
                />
              ))}
            </div>
          )}
        </div>

        {/* Input bar */}
        <div className="shrink-0 flex flex-col items-center">
          <div
            className="w-full pb-6 transition-all duration-500 ease-out"
            style={{
              maxWidth: isEmpty && !inputFocused ? '660px' : '100%',
              paddingLeft: isEmpty && !inputFocused ? '24px' : '80px',
              paddingRight: isEmpty && !inputFocused ? '24px' : '80px',
            }}
          >
          {/* Gradient border wrapper */}
          <div
            className="p-[1.5px] rounded-3xl transition-all duration-300"
            style={{
              background: "linear-gradient(135deg, rgba(124,58,237,0.5), rgba(59,130,246,0.5), rgba(167,139,250,0.3))",
              boxShadow: "0 8px 32px rgba(124,58,237,0.18), 0 2px 8px rgba(59,130,246,0.1)",
            }}
          >
            <div
              className="rounded-3xl px-5 py-4 transition-all duration-300"
              style={{
                background: "rgba(255,255,255,0.88)",
                backdropFilter: "blur(24px)",
                WebkitBackdropFilter: "blur(24px)",
              }}
              onFocus={(e) => {
                const wrapper = (e.currentTarget as HTMLElement).parentElement!;
                wrapper.style.background = "linear-gradient(135deg, rgba(124,58,237,0.85), rgba(59,130,246,0.85), rgba(167,139,250,0.6))";
                wrapper.style.boxShadow = "0 12px 40px rgba(124,58,237,0.28), 0 0 0 4px rgba(124,58,237,0.1)";
              }}
              onBlur={(e) => {
                const wrapper = (e.currentTarget as HTMLElement).parentElement!;
                wrapper.style.background = "linear-gradient(135deg, rgba(124,58,237,0.5), rgba(59,130,246,0.5), rgba(167,139,250,0.3))";
                wrapper.style.boxShadow = "0 8px 32px rgba(124,58,237,0.18), 0 2px 8px rgba(59,130,246,0.1)";
              }}
            >
              {/* URL tip */}
              {urlTip && (
                <div
                  className="flex items-center gap-2 mb-3 text-xs rounded-2xl px-3 py-2"
                  style={{
                    background: "rgba(59,130,246,0.08)",
                    border: "1px solid rgba(59,130,246,0.18)",
                    color: "#6B7280",
                  }}
                >
                  <Globe className="w-3.5 h-3.5 shrink-0" style={{ color: "#3B82F6" }} />
                  <span className="flex-1">Dán (Ctrl+V) đường link vào ô chat — link sẽ tự động được thêm làm tài liệu.</span>
                  <button onClick={() => setUrlTip(false)} className="text-gray-400 hover:text-gray-600 transition-colors">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}

              {/* Upload status chips */}
              {(attachments.length > 0 || recording) && (
                <div className="flex flex-wrap gap-2 mb-3">
                  {recording && (
                    <span
                      className="flex items-center gap-1.5 text-xs rounded-2xl px-3 py-1.5 font-medium"
                      style={{
                        background: "rgba(239,68,68,0.08)",
                        border: "1px solid rgba(239,68,68,0.2)",
                        color: "#EF4444",
                      }}
                    >
                      <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                      Đang ghi âm… bấm ⏹ để dừng & upload
                    </span>
                  )}
                  {attachments.map((a) => (
                    <span
                      key={a.id}
                      className="flex items-center gap-1.5 text-xs rounded-2xl px-3 py-1.5 max-w-[280px]"
                      style={
                        a.status === "error"
                          ? { background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#EF4444" }
                          : { background: "rgba(124,58,237,0.07)", border: "1px solid rgba(124,58,237,0.15)", color: "#1A1A2E" }
                      }
                    >
                      {a.status === "uploading" && <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" style={{ color: "#7C3AED" }} />}
                      {a.status === "done" && <CheckCircle2 className="w-3.5 h-3.5 shrink-0" style={{ color: "#10B981" }} />}
                      {a.status === "error" && <AlertCircle className="w-3.5 h-3.5 shrink-0" />}
                      <span className="truncate">{a.name}</span>
                      {a.detail && <span className="text-gray-400 shrink-0">· {a.detail}</span>}
                      {a.status !== "uploading" && (
                        <button
                          onClick={() => {
                            setAttachments((p) => p.filter((x) => x.id !== a.id));
                            const i = pendingDocsRef.current.indexOf(a.name);
                            if (i !== -1) pendingDocsRef.current.splice(i, 1);
                          }}
                          className="text-gray-400 hover:text-gray-600 shrink-0 transition-colors"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      )}
                    </span>
                  ))}
                </div>
              )}

              {/* Source picker popover */}
              {pickerOpen && (
                <div
                  className="mb-3 rounded-2xl px-3 py-3"
                  style={{
                    background: "rgba(255,255,255,0.9)",
                    border: "1px solid rgba(124,58,237,0.18)",
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-xs font-semibold text-[#1A1A2E]">Nguồn tham khảo cho câu hỏi</p>
                    <button onClick={() => setPickerOpen(false)} className="text-gray-400 hover:text-gray-600">
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <label className="flex items-center gap-2 text-xs text-[#1A1A2E] py-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useGlobalKb}
                      onChange={(e) => setUseGlobalKb(e.target.checked)}
                      className="accent-[#7C3AED]"
                    />
                    <BookOpen className="w-3.5 h-3.5" style={{ color: "#3B82F6" }} />
                    <span>Kho kiến thức chung <span className="text-gray-400">({kbDocCount} tài liệu)</span></span>
                  </label>
                  {sessionDocs.length > 0 ? (
                    <div className="max-h-40 overflow-y-auto mt-1 space-y-0.5">
                      {sessionDocs.map((d) => (
                        <label key={d.id} className="flex items-center gap-2 text-xs text-[#1A1A2E] py-1 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={!excludedDocs.has(d.id)}
                            onChange={(e) => {
                              setExcludedDocs((prev) => {
                                const next = new Set(prev);
                                if (e.target.checked) next.delete(d.id);
                                else next.add(d.id);
                                return next;
                              });
                            }}
                            className="accent-[#7C3AED]"
                          />
                          <FileText className="w-3.5 h-3.5 shrink-0" style={{ color: "#7C3AED" }} />
                          <span className="truncate flex-1">{d.name}</span>
                          <span className="text-gray-400 shrink-0">{d.chunk_count} chunks</span>
                        </label>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[11px] text-gray-400 mt-1">
                      Chưa có tài liệu nào trong cuộc trò chuyện này.
                    </p>
                  )}
                </div>
              )}

              {/* Main row: textarea + action buttons */}
              <div className="flex items-center gap-3">
                <textarea
                  ref={textareaRef}
                  className="flex-1 outline-none resize-none text-sm leading-6 bg-transparent placeholder-[#9CA3AF]"
                  placeholder="Ask me anything........"
                  style={{ color: "#1A1A2E" }}
                  rows={1}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKey}
                  onPaste={handlePaste}
                  onFocus={() => setInputFocused(true)}
                />

                {/* Action buttons */}
                <div className="flex items-center gap-1.5 shrink-0">
                  {/* Source picker button */}
                  <button
                    onClick={() => setPickerOpen((v) => !v)}
                    title="Chọn nguồn tham khảo cho câu hỏi"
                    className="relative w-9 h-9 rounded-2xl flex items-center justify-center transition-all duration-200"
                    style={
                      pickerOpen
                        ? {
                            background: "rgba(124,58,237,0.14)",
                            color: "#7C3AED",
                            border: "1px solid rgba(124,58,237,0.3)",
                          }
                        : {
                            background: "rgba(16,185,129,0.06)",
                            color: "#6EE7B7",
                            border: "1px solid rgba(16,185,129,0.12)",
                          }
                    }
                  >
                    <SlidersHorizontal className="w-4 h-4" />
                    {(excludedDocs.size > 0 || !useGlobalKb) && (
                      <span
                        className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full"
                        style={{ background: "#F59E0B", border: "2px solid #fff" }}
                        title="Đang lọc nguồn"
                      />
                    )}
                  </button>

                  {/* Mic button */}
                  <button
                    onClick={recording ? stopRecording : startRecording}
                    title={recording ? "Dừng ghi âm và upload" : "Ghi âm câu hỏi / ghi chú"}
                    className="w-9 h-9 rounded-2xl flex items-center justify-center transition-all duration-200"
                    style={
                      recording
                        ? {
                            background: "linear-gradient(135deg, #EF4444, #DC2626)",
                            color: "#fff",
                            boxShadow: "0 4px 12px rgba(239,68,68,0.4)",
                          }
                        : {
                            background: "rgba(124,58,237,0.06)",
                            color: "#A78BFA",
                            border: "1px solid rgba(124,58,237,0.12)",
                          }
                    }
                    onMouseEnter={(e) => {
                      if (!recording) {
                        (e.currentTarget as HTMLElement).style.background = "rgba(124,58,237,0.12)";
                        (e.currentTarget as HTMLElement).style.color = "#7C3AED";
                        (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 12px rgba(124,58,237,0.2)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!recording) {
                        (e.currentTarget as HTMLElement).style.background = "rgba(124,58,237,0.06)";
                        (e.currentTarget as HTMLElement).style.color = "#A78BFA";
                        (e.currentTarget as HTMLElement).style.boxShadow = "none";
                      }
                    }}
                  >
                    {recording ? <Square className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                  </button>

                  {/* Attach button */}
                  <button
                    onClick={() => openPicker(FILE_ACCEPT)}
                    title="Đính kèm tài liệu (PDF, TXT, ảnh, audio)"
                    className="w-9 h-9 rounded-2xl flex items-center justify-center transition-all duration-200"
                    style={{
                      background: "rgba(59,130,246,0.06)",
                      color: "#93C5FD",
                      border: "1px solid rgba(59,130,246,0.12)",
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.background = "rgba(59,130,246,0.12)";
                      (e.currentTarget as HTMLElement).style.color = "#3B82F6";
                      (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 12px rgba(59,130,246,0.2)";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.background = "rgba(59,130,246,0.06)";
                      (e.currentTarget as HTMLElement).style.color = "#93C5FD";
                      (e.currentTarget as HTMLElement).style.boxShadow = "none";
                    }}
                  >
                    <Paperclip className="w-4 h-4" />
                  </button>

                  {/* Send / Stop button */}
                  <button
                    onClick={sending ? stopStreaming : send}
                    disabled={!sending && !input.trim()}
                    title={sending ? "Dừng tạo câu trả lời" : "Gửi"}
                    className="h-9 px-4 rounded-2xl flex items-center gap-2 text-white text-xs font-semibold transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                    style={{
                      background: sending
                        ? "linear-gradient(135deg, #EF4444, #DC2626)"
                        : "linear-gradient(135deg, #7C3AED, #3B82F6)",
                      boxShadow: sending
                        ? "0 4px 16px rgba(239,68,68,0.45)"
                        : "0 4px 16px rgba(124,58,237,0.45)",
                      minWidth: "2.25rem",
                    }}
                    onMouseEnter={(e) => {
                      if (!sending && input.trim()) {
                        (e.currentTarget as HTMLElement).style.background = "linear-gradient(135deg, #6D28D9, #2563EB)";
                        (e.currentTarget as HTMLElement).style.boxShadow = "0 6px 20px rgba(124,58,237,0.55)";
                        (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.background = sending
                        ? "linear-gradient(135deg, #EF4444, #DC2626)"
                        : "linear-gradient(135deg, #7C3AED, #3B82F6)";
                      (e.currentTarget as HTMLElement).style.boxShadow = sending
                        ? "0 4px 16px rgba(239,68,68,0.45)"
                        : "0 4px 16px rgba(124,58,237,0.45)";
                      (e.currentTarget as HTMLElement).style.transform = "";
                    }}
                  >
                    {sending ? <Square className="w-4 h-4" /> : <Send className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </div>
          </div>
          <p className="text-center text-[10px] mt-3" style={{ color: "#C4B5FD" }}>
            IT Knowledge Assistant • Powered by Modular RAG
          </p>
          </div>{/* end px-20 pb-6 */}

          {/* ── Animated spacer: lifts input toward center when new chat ── */}
          <div
            aria-hidden="true"
            style={{
              height: isEmpty && !inputFocused ? '36vh' : '0px',
              transition: 'height 0.55s cubic-bezier(0.4, 0, 0.2, 1)',
              flexShrink: 0,
              overflow: 'hidden',
            }}
          />
        </div>{/* end shrink-0 flex flex-col */}
      </div>

      {/* Document viewer (citation click-through) — fixed side drawer */}
      {viewerTarget && (
        <>
          {/* Semi-transparent backdrop: click to close */}
          <div
            aria-label="Đóng tài liệu"
            onClick={() => setViewerTarget(null)}
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(15,10,30,0.22)",
              backdropFilter: "blur(2px)",
              WebkitBackdropFilter: "blur(2px)",
              zIndex: 40,
              animation: "fadeIn 0.2s ease",
            }}
          />
          {/* Drawer panel */}
          <div
            style={{
              position: "fixed",
              top: 0,
              right: 0,
              bottom: 0,
              width: "min(44%, 560px)",
              zIndex: 41,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
              animation: "slideInRight 0.28s cubic-bezier(0.4,0,0.2,1)",
            }}
          >
            <DocViewerPanel target={viewerTarget} onClose={() => setViewerTarget(null)} />
          </div>
        </>
      )}
    </div>
  );
}

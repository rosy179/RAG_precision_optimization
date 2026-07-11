import { useState, useEffect, useRef, useCallback } from "react";
import {
  Send, Mic, Paperclip, X, Globe,
  Loader2, Square, CheckCircle2, AlertCircle,
} from "lucide-react";
import { sessionsAPI, documentsAPI } from "../api/client";
import MessageBubble from "../components/MessageBubble";
import AiRobotIcon from "../components/AiRobotIcon";
import type { Message } from "../components/MessageBubble";

interface Props { sessionId: string | null; onSessionCreated: (id: string) => void; }

interface Attachment {
  id: string;
  name: string;
  status: "uploading" | "done" | "error";
  detail?: string;
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
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const sessionIdRef = useRef<string | null>(sessionId);
  const pendingDocsRef = useRef<string[]>([]);

  useEffect(() => {
    if (sessionIdRef.current !== sessionId) {
      pendingDocsRef.current = [];
      setAttachments([]);
      setInputFocused(false);
    }
    sessionIdRef.current = sessionId;
  }, [sessionId]);

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
    // Chỉ cuộn trong khung tin nhắn (scrollIntoView cuộn cả các container cha
    // làm lệch toàn bộ layout). Mở session cũ thì nhảy thẳng xuống cuối,
    // tin nhắn mới thêm vào thì cuộn mượt.
    const appended = messages.length === prevCountRef.current + 1;
    prevCountRef.current = messages.length;
    el.scrollTo({ top: el.scrollHeight, behavior: appended ? "smooth" : "auto" });
  }, [messages, sending]);

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
  }, [ensureSession]);

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
    } catch (e: any) {
      setAttachments((p) => p.map((a) =>
        a.id === id ? { ...a, status: "error" as const, detail: e?.response?.data?.detail || "Không thể fetch URL" } : a
      ));
    }
  }, [ensureSession]);

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

  const send = useCallback(async () => {
    const q = input.trim();
    if (!q || sending) return;

    const isNewSession = !sessionIdRef.current;
    const sid = await ensureSession(false);

    const attached = pendingDocsRef.current;
    pendingDocsRef.current = [];
    setAttachments((p) => p.filter((a) => a.status !== "done"));

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: q, attachments: attached };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);

    const history = messages.slice(-6).map((m) => ({ role: m.role, content: m.content }));
    try {
      const data = await sessionsAPI.chat(sid, q, history, attached);
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, aiMsg]);
      if (isNewSession) onSessionCreated(sid);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setMessages((prev) => [...prev, {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: detail || "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.",
      }]);
      if (isNewSession) onSessionCreated(sid);
    } finally {
      setSending(false);
    }
  }, [input, sending, messages, ensureSession, onSessionCreated]);

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
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
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
            <div className="px-20 py-6">
              {messages.map((m) => <MessageBubble key={m.id} msg={m} />)}
              {sending && (
                <div className="flex items-center gap-3 mb-4">
                  <AiRobotIcon mini />
                  <div
                    className="rounded-3xl rounded-tl-md px-5 py-4"
                    style={{
                      background: "rgba(255,255,255,0.72)",
                      backdropFilter: "blur(16px)",
                      border: "1px solid rgba(255,255,255,0.85)",
                      boxShadow: "0 4px 24px rgba(124,58,237,0.08)",
                    }}
                  >
                    <div className="flex items-center gap-1.5">
                      {[0, 150, 300].map((delay) => (
                        <div
                          key={delay}
                          className="w-2 h-2 rounded-full"
                          style={{
                            background: "linear-gradient(135deg, #7C3AED, #3B82F6)",
                            animation: `dot-bounce 1.2s ease-in-out infinite`,
                            animationDelay: `${delay}ms`,
                          }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              )}
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

                  {/* Send button */}
                  <button
                    onClick={send}
                    disabled={!input.trim() || sending}
                    className="h-9 px-4 rounded-2xl flex items-center gap-2 text-white text-xs font-semibold transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                    style={{
                      background: "linear-gradient(135deg, #7C3AED, #3B82F6)",
                      boxShadow: "0 4px 16px rgba(124,58,237,0.45)",
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
                      (e.currentTarget as HTMLElement).style.background = "linear-gradient(135deg, #7C3AED, #3B82F6)";
                      (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 16px rgba(124,58,237,0.45)";
                      (e.currentTarget as HTMLElement).style.transform = "";
                    }}
                  >
                    {sending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
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
    </div>
  );
}

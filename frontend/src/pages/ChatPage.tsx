import { useState, useEffect, useRef, useCallback } from "react";
import {
  Send, Mic, Paperclip, X, Bot, FileText, Globe, Image as ImageIcon,
  Loader2, ChevronRight, Square, CheckCircle2, AlertCircle,
} from "lucide-react";
import { sessionsAPI, documentsAPI } from "../api/client";
import MessageBubble from "../components/MessageBubble";
import type { Message } from "../components/MessageBubble";
import DocumentPanel from "../components/DocumentPanel";

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

// ~6 lines of text-sm (20px line-height); beyond this the textarea scrolls.
const MAX_INPUT_HEIGHT = 120;

export default function ChatPage({ sessionId, onSessionCreated }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  const [docsKey, setDocsKey] = useState(0);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [urlSuggestion, setUrlSuggestion] = useState<string | null>(null);
  const [addingUrl, setAddingUrl] = useState(false);
  const [urlTip, setUrlTip] = useState(false);
  const [recording, setRecording] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);

  useEffect(() => {
    if (!sessionId) { setMessages([]); return; }
    sessionsAPI.messages(sessionId).then(setMessages).catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-grow the textarea with its content, up to MAX_INPUT_HEIGHT,
  // then scroll. Runs on every input change, including programmatic
  // clears (after send / after adding a pasted URL as a document).
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_INPUT_HEIGHT)}px`;
    el.style.overflowY = el.scrollHeight > MAX_INPUT_HEIGHT ? "auto" : "hidden";
  }, [input]);

  // ── Upload from input bar ─────────────────────────────
  const uploadFiles = useCallback(async (files: File[]) => {
    for (const file of files) {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      setAttachments((p) => [...p, { id, name: file.name, status: "uploading" }]);
      try {
        const res = await documentsAPI.uploadFile(file);
        setAttachments((p) => p.map((a) =>
          a.id === id ? { ...a, status: "done" as const, detail: `${res.chunk_count} chunks` } : a
        ));
        setDocsKey((k) => k + 1);
        setTimeout(() => setAttachments((p) => p.filter((a) => a.id !== id)), 5000);
      } catch (e: any) {
        setAttachments((p) => p.map((a) =>
          a.id === id ? { ...a, status: "error" as const, detail: e?.response?.data?.detail || "Lỗi upload" } : a
        ));
      }
    }
  }, []);

  const addUrlAsDocument = useCallback(async (url: string) => {
    setAddingUrl(true);
    const id = `${Date.now()}-url`;
    setAttachments((p) => [...p, { id, name: url, status: "uploading" }]);
    try {
      const res = await documentsAPI.uploadUrl(url);
      setAttachments((p) => p.map((a) =>
        a.id === id ? { ...a, name: res.name || url, status: "done" as const, detail: `${res.chunk_count} chunks` } : a
      ));
      setDocsKey((k) => k + 1);
      setTimeout(() => setAttachments((p) => p.filter((a) => a.id !== id)), 5000);
      // If the input only contains this URL, clear it — it's now a document.
      setInput((prev) => (prev.trim() === url ? "" : prev));
    } catch (e: any) {
      setAttachments((p) => p.map((a) =>
        a.id === id ? { ...a, status: "error" as const, detail: e?.response?.data?.detail || "Không thể fetch URL" } : a
      ));
    } finally {
      setAddingUrl(false);
      setUrlSuggestion(null);
    }
  }, []);

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
      setUrlSuggestion(text);
      setUrlTip(false);
    }
  }, [uploadFiles]);

  // ── Voice recording ───────────────────────────────────
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

    let sid = sessionId;
    const isNewSession = !sid;
    if (!sid) {
      const s = await sessionsAPI.create();
      sid = s.session_id;
    }

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);

    const history = messages.slice(-6).map((m) => ({ role: m.role, content: m.content }));
    try {
      const data = await sessionsAPI.chat(sid!, q, history);
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, aiMsg]);
      // Only switch the active session (and trigger the sidebar/messages refetch)
      // after the DB has the full exchange saved, otherwise the refetch can
      // race the chat request and briefly wipe the user's own message.
      if (isNewSession) onSessionCreated(sid!);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setMessages((prev) => [...prev, {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: detail || "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.",
      }]);
      if (isNewSession) onSessionCreated(sid!);
    } finally {
      setSending(false);
    }
  }, [input, sending, sessionId, messages, onSessionCreated]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const isEmpty = messages.length === 0;

  const shortcuts = [
    {
      icon: <FileText className="w-6 h-6 text-[#7C3AED]" />, label: "Chat Files", bg: "#EDE9FE",
      onClick: () => openPicker(DOC_ACCEPT),
    },
    {
      icon: <ImageIcon className="w-6 h-6 text-purple-400" />, label: "Images", bg: "#F5F3FF",
      onClick: () => openPicker(IMAGE_ACCEPT),
    },
    {
      icon: <Globe className="w-6 h-6 text-[#3B82F6]" />, label: "Web URL", bg: "#EFF6FF",
      onClick: () => { setUrlTip(true); textareaRef.current?.focus(); },
    },
    {
      icon: <Mic className="w-6 h-6 text-indigo-400" />, label: "Audio", bg: "#EEF2FF",
      onClick: () => openPicker(AUDIO_ACCEPT),
    },
  ];

  return (
    <div className="flex flex-1 min-h-screen bg-white overflow-hidden">
      {/* Hidden file input shared by 📎 button and shortcut cards */}
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
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#E8E0FF] bg-white shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-[#EDE9FE] flex items-center justify-center">
              <Bot className="w-4 h-4 text-[#7C3AED]" />
            </div>
            <span className="font-semibold text-[#1A1A2E]">
              {isEmpty ? "New Chat" : messages[0]?.content?.slice(0, 40) + "…"}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowDocs(!showDocs)}
              className="flex items-center gap-1.5 text-xs font-medium text-[#7C3AED] hover:bg-[#EDE9FE] px-3 py-1.5 rounded-xl transition-all"
            >
              <FileText className="w-3.5 h-3.5" />
              Tài liệu
            </button>
            <button className="w-8 h-8 rounded-xl flex items-center justify-center text-gray-400 hover:bg-gray-100 transition-all">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {isEmpty ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#7C3AED] to-[#3B82F6] flex items-center justify-center shadow-lg mb-5">
                <Bot className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-[#1A1A2E] mb-1">
                Xin chào! 👋
              </h2>
              <p className="text-gray-500 text-sm mb-8">
                Tôi có thể giúp gì cho bạn?<br />
                Đính kèm tệp, dán ảnh/URL hoặc ghi âm ngay tại ô chat.
              </p>

              {/* Shortcut cards */}
              <div className="grid grid-cols-4 gap-3">
                {shortcuts.map((s) => (
                  <button
                    key={s.label}
                    onClick={s.onClick}
                    className="flex flex-col items-center gap-2 bg-white border border-[#E0D9FF] rounded-2xl px-4 py-4 cursor-pointer hover:shadow-card-hover hover:border-[#7C3AED]/40 transition-all"
                  >
                    <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: s.bg }}>
                      {s.icon}
                    </div>
                    <span className="text-xs font-medium text-[#1A1A2E]">{s.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((m) => <MessageBubble key={m.id} msg={m} />)}
              {sending && (
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-9 h-9 rounded-full bg-gradient-to-br from-[#7C3AED] to-[#3B82F6] flex items-center justify-center shadow">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <div className="bg-white rounded-3xl rounded-tl-md px-5 py-4 shadow-card">
                    <div className="flex items-center gap-1.5">
                      <div className="w-2 h-2 bg-[#7C3AED] rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                      <div className="w-2 h-2 bg-[#7C3AED] rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                      <div className="w-2 h-2 bg-[#7C3AED] rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </>
          )}
        </div>

        {/* Input bar */}
        <div className="px-6 pb-6 shrink-0">
          <div className="bg-white border border-[#E0D9FF] rounded-2xl px-4 py-3 shadow-card focus-within:border-[#7C3AED] focus-within:ring-2 focus-within:ring-[#7C3AED]/20 transition-all">
            {/* URL tip (from Web URL shortcut) */}
            {urlTip && !urlSuggestion && (
              <div className="flex items-center gap-2 mb-2 text-xs text-gray-500 bg-[#EFF6FF] rounded-xl px-3 py-2">
                <Globe className="w-3.5 h-3.5 text-[#3B82F6] shrink-0" />
                <span className="flex-1">Dán (Ctrl+V) đường link vào ô chat để thêm làm tài liệu.</span>
                <button onClick={() => setUrlTip(false)} className="text-gray-400 hover:text-gray-600">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* URL detected → offer to index it */}
            {urlSuggestion && (
              <div className="flex items-center gap-2 mb-2 text-xs bg-[#EFF6FF] rounded-xl px-3 py-2">
                <Globe className="w-3.5 h-3.5 text-[#3B82F6] shrink-0" />
                <span className="flex-1 truncate text-[#1A1A2E]">{urlSuggestion}</span>
                <button
                  onClick={() => addUrlAsDocument(urlSuggestion)}
                  disabled={addingUrl}
                  className="btn-primary px-2.5 py-1 text-[11px] shrink-0"
                >
                  {addingUrl ? <Loader2 className="w-3 h-3 animate-spin" /> : "Thêm làm tài liệu"}
                </button>
                <button onClick={() => setUrlSuggestion(null)} className="text-gray-400 hover:text-gray-600 shrink-0">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Upload status chips */}
            {(attachments.length > 0 || recording) && (
              <div className="flex flex-wrap gap-2 mb-2">
                {recording && (
                  <span className="flex items-center gap-1.5 text-xs bg-red-50 text-red-600 border border-red-200 rounded-xl px-3 py-1.5">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    Đang ghi âm… bấm ⏹ để dừng &amp; upload
                  </span>
                )}
                {attachments.map((a) => (
                  <span
                    key={a.id}
                    className={`flex items-center gap-1.5 text-xs rounded-xl px-3 py-1.5 border max-w-[280px] ${
                      a.status === "error"
                        ? "bg-red-50 text-red-600 border-red-200"
                        : "bg-[#F5F3FF] text-[#1A1A2E] border-[#E0D9FF]"
                    }`}
                  >
                    {a.status === "uploading" && <Loader2 className="w-3.5 h-3.5 animate-spin text-[#7C3AED] shrink-0" />}
                    {a.status === "done" && <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0" />}
                    {a.status === "error" && <AlertCircle className="w-3.5 h-3.5 shrink-0" />}
                    <span className="truncate">{a.name}</span>
                    {a.detail && <span className="text-gray-400 shrink-0">· {a.detail}</span>}
                    {a.status !== "uploading" && (
                      <button
                        onClick={() => setAttachments((p) => p.filter((x) => x.id !== a.id))}
                        className="text-gray-400 hover:text-gray-600 shrink-0"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    )}
                  </span>
                ))}
              </div>
            )}

            <div className="flex items-end gap-3">
              <textarea
                ref={textareaRef}
                className="flex-1 outline-none resize-none text-sm leading-5 text-[#1A1A2E] placeholder:text-gray-400 bg-transparent"
                placeholder="Hỏi về tài liệu, hoặc dán ảnh / URL vào đây..."
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKey}
                onPaste={handlePaste}
              />
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={recording ? stopRecording : startRecording}
                  title={recording ? "Dừng ghi âm và upload" : "Ghi âm câu hỏi / ghi chú"}
                  className={`w-8 h-8 rounded-xl flex items-center justify-center transition-all ${
                    recording
                      ? "bg-red-500 text-white hover:bg-red-600 animate-pulse"
                      : "text-gray-400 hover:text-[#7C3AED] hover:bg-[#EDE9FE]"
                  }`}
                >
                  {recording ? <Square className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => openPicker(FILE_ACCEPT)}
                  title="Đính kèm tài liệu (PDF, TXT, ảnh, audio)"
                  className="w-8 h-8 rounded-xl flex items-center justify-center text-gray-400 hover:text-[#7C3AED] hover:bg-[#EDE9FE] transition-all"
                >
                  <Paperclip className="w-4 h-4" />
                </button>
                <button
                  onClick={send}
                  disabled={!input.trim() || sending}
                  className="w-9 h-9 rounded-xl bg-[#7C3AED] hover:bg-[#6D28D9] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center text-white shadow transition-all"
                >
                  {sending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
          <p className="text-center text-[10px] text-gray-400 mt-2">
            IT Knowledge Assistant • Powered by Modular RAG
          </p>
        </div>
      </div>

      {/* Document side panel — read-only list of uploaded docs */}
      {showDocs && (
        <div className="w-[300px] border-l border-[#E0D9FF] bg-[#FAFAFA] shrink-0 flex flex-col">
          <div className="flex items-center justify-between px-4 py-4 border-b border-[#E0D9FF]">
            <h3 className="font-semibold text-sm text-[#1A1A2E]">Tài liệu</h3>
            <button
              onClick={() => setShowDocs(false)}
              className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-400 hover:bg-gray-100 transition-all"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <DocumentPanel
              key={docsKey}
              onDocsChange={() => setDocsKey((k) => k + 1)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

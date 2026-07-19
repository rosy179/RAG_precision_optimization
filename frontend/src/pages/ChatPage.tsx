import { useState, useEffect, useRef, useCallback } from "react";
import {
  Send, Mic, Paperclip, X, Globe,
  Loader2, Square, CheckCircle2, AlertCircle,
  Download, SlidersHorizontal,
} from "lucide-react";
import { sessionsAPI, documentsAPI, knowledgeAPI } from "../api/client";
import MessageBubble from "../components/MessageBubble";
import AiRobotIcon from "../components/AiRobotIcon";
import DocViewerPanel from "../components/DocViewerPanel";
import SourcePicker from "../components/SourcePicker";
import { useChatStream } from "../hooks/useChatStream";
import type { ViewerTarget } from "../components/DocViewerPanel";
import type { AttachmentDoc, Message, Source } from "../components/MessageBubble";

interface Props { sessionId: string | null; onSessionCreated: (id: string) => void; }

interface Attachment {
  id: string;
  name: string;
  status: "uploading" | "done" | "error";
  detail?: string;
  /** Local object URL preview — set for image files */
  previewUrl?: string;
}

interface SessionDoc {
  id: string;
  name: string;
  type: string;
  chunk_count: number;
}

const DOC_ACCEPT = ".pdf,.txt,.md,.markdown,.docx,.pptx,.xlsx";
const IMAGE_ACCEPT = ".png,.jpg,.jpeg,.webp";
const AUDIO_ACCEPT = ".mp3,.wav,.m4a,.ogg,.webm";
const FILE_ACCEPT = `${DOC_ACCEPT},${IMAGE_ACCEPT},${AUDIO_ACCEPT}`;

const URL_REGEX = /^https?:\/\/\S+$/i;
const MAX_INPUT_HEIGHT = 120;

export default function ChatPage({ sessionId, onSessionCreated }: Props) {
  // Message list + SSE streaming plumbing live in a dedicated hook (E3).
  const { messages, setMessages, sending, setSending, patchMessage, runStream, stopStreaming } =
    useChatStream(onSessionCreated);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [urlTip, setUrlTip] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
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
  // Live dictation (Web Speech API): base = input when dictation started,
  // final = committed transcript; interim text is re-rendered on top of both.
  const recognitionRef = useRef<any>(null);
  const recordingRef = useRef(false);
  const voiceBaseRef = useRef("");
  const voiceFinalRef = useRef("");
  const sessionIdRef = useRef<string | null>(sessionId);
  const pendingDocsRef = useRef<string[]>([]);
  // Uploads still in flight — send() waits for them so a question typed right
  // after pasting a link/file actually sees that newest document.
  const inflightUploadsRef = useRef<Set<Promise<unknown>>>(new Set());

  useEffect(() => {
    if (sessionIdRef.current !== sessionId) {
      pendingDocsRef.current = [];
      prevCountRef.current = 0; // reset so bulk-load scroll fires correctly
      setAttachments((p) => {
        p.forEach((a) => a.previewUrl && URL.revokeObjectURL(a.previewUrl));
        return [];
      });
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
      const previewUrl = file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined;
      setAttachments((p) => [...p, { id, name: file.name, status: "uploading", previewUrl }]);
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

  /** Register an upload so send() can wait for it (upload flows never reject
   *  — errors surface as attachment chips). */
  const trackUpload = useCallback((p: Promise<unknown>) => {
    inflightUploadsRef.current.add(p);
    p.finally(() => inflightUploadsRef.current.delete(p));
    return p;
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
      trackUpload(uploadFiles(files));
      return;
    }
    const text = e.clipboardData.getData("text").trim();
    if (URL_REGEX.test(text)) {
      e.preventDefault();
      setUrlTip(false);
      trackUpload(addUrlAsDocument(text));
    }
  }, [uploadFiles, addUrlAsDocument, trackUpload]);

  const micError = useCallback((detail: string) => {
    setAttachments((p) => [...p, {
      id: `${Date.now()}-mic`, name: "Micro", status: "error", detail,
    }]);
  }, []);

  /** Voice input: live speech-to-text into the input box (Web Speech API).
   *  Browsers without it (e.g. Firefox) record audio and transcribe it
   *  server-side with Whisper — the text lands in the input box either way. */
  const startRecording = useCallback(async () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SR) {
      const rec = new SR();
      rec.lang = "vi-VN";
      rec.continuous = true;
      rec.interimResults = true;
      voiceBaseRef.current = input ? input.replace(/\s+$/, "") + " " : "";
      voiceFinalRef.current = "";
      rec.onresult = (e: any) => {
        // Late results after stop (e.g. right after Send cleared the input)
        // must not resurrect the dictated text.
        if (!recordingRef.current) return;
        let interim = "";
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const t = e.results[i][0].transcript;
          if (e.results[i].isFinal) voiceFinalRef.current += t.trim() + " ";
          else interim += t;
        }
        setInput(voiceBaseRef.current + voiceFinalRef.current + interim);
      };
      rec.onerror = (e: any) => {
        if (e.error === "not-allowed" || e.error === "service-not-allowed") {
          recordingRef.current = false;
          recognitionRef.current = null;
          setRecording(false);
          micError("Không truy cập được micro");
        }
        // "no-speech" / "aborted": onend restarts while still recording
      };
      rec.onend = () => {
        // Chrome stops after a silence gap — keep listening until ⏹ is pressed
        if (recordingRef.current && recognitionRef.current === rec) {
          try { rec.start(); } catch { /* already stopped */ }
        }
      };
      try {
        rec.start();
        recognitionRef.current = rec;
        recordingRef.current = true;
        setRecording(true);
        textareaRef.current?.focus();
      } catch {
        micError("Không khởi động được nhận dạng giọng nói");
      }
      return;
    }

    // Fallback: record, then Whisper transcription fills the input box
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const parts: Blob[] = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) parts.push(e.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const mime = recorder.mimeType || "audio/webm";
        const ext = mime.includes("ogg") ? "ogg" : "webm";
        const file = new File(parts, `voice.${ext}`, { type: mime });
        setTranscribing(true);
        try {
          const res = await documentsAPI.transcribe(file);
          const text = (res.text || "").trim();
          if (text) setInput((prev) => (prev ? prev.replace(/\s+$/, "") + " " : "") + text);
        } catch (e: any) {
          micError(e?.response?.data?.detail || "Không chuyển được giọng nói thành văn bản");
        } finally {
          setTranscribing(false);
          textareaRef.current?.focus();
        }
      };
      recorder.start();
      recorderRef.current = recorder;
      recordingRef.current = true;
      setRecording(true);
    } catch {
      micError("Không truy cập được micro");
    }
  }, [input, micError]);

  const stopRecording = useCallback(() => {
    recordingRef.current = false;
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    recorderRef.current?.stop();
    recorderRef.current = null;
    setRecording(false);
    textareaRef.current?.focus();
  }, []);

  /** Chat-body fields reflecting the source picker state. */
  const sourceParams = useCallback(() => ({
    ...(excludedDocs.size > 0
      ? { include_doc_ids: sessionDocs.filter((d) => !excludedDocs.has(d.id)).map((d) => d.id) }
      : {}),
    use_global_kb: useGlobalKb,
  }), [excludedDocs, sessionDocs, useGlobalKb]);

  // Attachment names in messages → session doc (for thumbnails & click-to-open)
  const resolveAttachment = useCallback(
    (name: string) => sessionDocs.find((d) => d.name === name),
    [sessionDocs],
  );

  const openAttachment = useCallback((doc: AttachmentDoc) => {
    setViewerTarget({ docId: doc.id, scope: "session", title: doc.name });
  }, []);

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
    if (recordingRef.current) stopRecording();
    setSending(true);

    // A question sent right after pasting a link / picking a file must wait
    // for that upload: otherwise the newest document is neither indexed nor
    // attached yet, and retrieval answers from the OLDER session documents.
    while (inflightUploadsRef.current.size > 0) {
      await Promise.allSettled(Array.from(inflightUploadsRef.current));
    }

    const isNewSession = !sessionIdRef.current;
    let sid: string;
    try {
      sid = await ensureSession(false);
    } catch {
      setSending(false);
      return;
    }

    const attached = pendingDocsRef.current;
    pendingDocsRef.current = [];
    setAttachments((p) => {
      p.forEach((a) => a.status === "done" && a.previewUrl && URL.revokeObjectURL(a.previewUrl));
      return p.filter((a) => a.status !== "done");
    });

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: q, attachments: attached };
    const aiId = (Date.now() + 1).toString();
    setMessages((prev) => [...prev, userMsg, { id: aiId, role: "assistant", content: "", streaming: true }]);
    setInput("");

    const history = messages.slice(-6).map((m) => ({ role: m.role, content: m.content }));
    await runStream(sid, aiId, { question: q, history, attachments: attached, ...sourceParams() });
    if (isNewSession) onSessionCreated(sid);
  }, [input, sending, messages, ensureSession, onSessionCreated, runStream, sourceParams, stopRecording]);

  /** Ask a suggested follow-up chip: a plain question in the existing session,
   *  no attachments, current answer as history. */
  const askSuggestion = useCallback(async (q: string) => {
    if (sending || !q.trim() || !sessionIdRef.current) return;
    setSending(true);
    const sid = sessionIdRef.current;
    const userMsg: Message = { id: Date.now().toString(), role: "user", content: q };
    const aiId = (Date.now() + 1).toString();
    setMessages((prev) => [...prev, userMsg, { id: aiId, role: "assistant", content: "", streaming: true }]);
    const history = messages.slice(-6).map((m) => ({ role: m.role, content: m.content }));
    await runStream(sid, aiId, { question: q, history, ...sourceParams() });
  }, [sending, messages, runStream, sourceParams]);

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

    patchMessage(aiMsg.id, { content: "", sources: undefined, feedback: null, steps: [], streaming: true });
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

  /** Export the conversation to a Markdown file, citations preserved and each
   *  answer's sources appended as a numbered list. Fully client-side. */
  const exportConversation = useCallback(() => {
    if (messages.length === 0) return;
    const firstQ = messages.find((m) => m.role === "user")?.content ?? "Hội thoại";
    const title = firstQ.length > 60 ? firstQ.slice(0, 60) + "…" : firstQ;
    const lines: string[] = [
      `# ${title}`,
      "",
      `_Xuất từ RAG Chatbot · ${new Date().toLocaleString("vi-VN")}_`,
      "",
    ];
    for (const m of messages) {
      if (!m.content) continue;
      if (m.role === "user") {
        lines.push(`## 👤 ${m.content}`, "");
      } else {
        lines.push("**🤖 Trả lời:**", "", m.content, "");
        if (m.sources && m.sources.length > 0) {
          lines.push("**Nguồn tham khảo:**");
          m.sources.forEach((s) => {
            const scope = s.scope === "global" ? " _(kho chung)_" : "";
            lines.push(`${s.rank}. ${s.title}${s.page ? ` — tr.${s.page}` : ""}${scope}`);
          });
          lines.push("");
        }
        lines.push("---", "");
      }
    }
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    a.href = url;
    a.download = `hoi-thoai-${stamp}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }, [messages]);

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
          if (files.length) trackUpload(uploadFiles(files));
          e.target.value = "";
        }}
      />

      {/* Main chat */}
      <div className="flex flex-col flex-1 min-w-0 relative">

        {/* Export conversation — floating, appears once there are messages */}
        {!isEmpty && (
          <button
            onClick={exportConversation}
            title="Xuất hội thoại ra Markdown"
            className="absolute top-3 right-3 z-20 h-9 px-3 rounded-2xl flex items-center gap-1.5 text-xs font-medium shadow-sm transition-all hover:shadow-md hover:-translate-y-px"
            style={{
              background: "rgba(255,255,255,0.8)",
              border: "1px solid rgba(124,58,237,0.2)",
              backdropFilter: "blur(10px)",
              color: "#7C3AED",
            }}
          >
            <Download className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Xuất</span>
          </button>
        )}

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
            <div className="px-4 sm:px-8 md:px-12 lg:px-20 pt-16 md:pt-6 pb-16">
              {messages.map((m, i) => (
                <MessageBubble
                  key={m.id}
                  msg={m}
                  onFeedback={handleFeedback}
                  onOpenSource={openSource}
                  resolveAttachment={resolveAttachment}
                  onOpenAttachment={openAttachment}
                  onRegenerate={
                    !sending && m.role === "assistant" && i === messages.length - 1
                      ? regenerate
                      : undefined
                  }
                  onAskSuggestion={
                    !sending && m.role === "assistant" && i === messages.length - 1
                      ? askSuggestion
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
            className={`w-full pb-4 md:pb-6 transition-all duration-500 ease-out ${
              isEmpty && !inputFocused
                ? "max-w-[660px] px-4 sm:px-6"
                : "max-w-full px-3 sm:px-8 md:px-12 lg:px-20"
            }`}
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
              className="rounded-3xl px-3.5 py-3 sm:px-5 sm:py-4 transition-all duration-300"
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
              {(attachments.length > 0 || recording || transcribing) && (
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
                      Đang nghe… cứ nói, chữ sẽ hiện trong ô nhập — bấm ⏹ để dừng
                    </span>
                  )}
                  {transcribing && (
                    <span
                      className="flex items-center gap-1.5 text-xs rounded-2xl px-3 py-1.5 font-medium"
                      style={{
                        background: "rgba(124,58,237,0.07)",
                        border: "1px solid rgba(124,58,237,0.15)",
                        color: "#7C3AED",
                      }}
                    >
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      Đang chuyển giọng nói thành văn bản…
                    </span>
                  )}
                  {attachments.map((a) => {
                    const remove = () => {
                      if (a.previewUrl) URL.revokeObjectURL(a.previewUrl);
                      setAttachments((p) => p.filter((x) => x.id !== a.id));
                      const i = pendingDocsRef.current.indexOf(a.name);
                      if (i !== -1) pendingDocsRef.current.splice(i, 1);
                    };
                    // Image attachment → mini thumbnail with status overlay
                    if (a.previewUrl) {
                      return (
                        <span key={a.id} className="relative inline-block" title={a.name}>
                          <img
                            src={a.previewUrl}
                            alt={a.name}
                            className="h-14 w-14 object-cover rounded-2xl block"
                            style={{
                              border: a.status === "error"
                                ? "1.5px solid rgba(239,68,68,0.5)"
                                : "1.5px solid rgba(124,58,237,0.25)",
                              opacity: a.status === "uploading" ? 0.6 : 1,
                            }}
                          />
                          {a.status === "uploading" && (
                            <span className="absolute inset-0 flex items-center justify-center">
                              <Loader2 className="w-4 h-4 animate-spin" style={{ color: "#7C3AED" }} />
                            </span>
                          )}
                          {a.status === "error" && (
                            <span className="absolute inset-0 flex items-center justify-center rounded-2xl" style={{ background: "rgba(239,68,68,0.15)" }}>
                              <AlertCircle className="w-4 h-4" style={{ color: "#EF4444" }} />
                            </span>
                          )}
                          {a.status !== "uploading" && (
                            <button
                              onClick={remove}
                              className="absolute -top-1.5 -right-1.5 rounded-full flex items-center justify-center shadow transition-colors"
                              style={{ background: "#fff", border: "1px solid rgba(0,0,0,0.08)", width: 18, height: 18 }}
                            >
                              <X className="w-3 h-3 text-gray-500 hover:text-gray-700" />
                            </button>
                          )}
                        </span>
                      );
                    }
                    return (
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
                            onClick={remove}
                            className="text-gray-400 hover:text-gray-600 shrink-0 transition-colors"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        )}
                      </span>
                    );
                  })}
                </div>
              )}

              {/* Source picker popover */}
              {pickerOpen && (
                <SourcePicker
                  sessionDocs={sessionDocs}
                  kbDocCount={kbDocCount}
                  excludedDocs={excludedDocs}
                  setExcludedDocs={setExcludedDocs}
                  useGlobalKb={useGlobalKb}
                  setUseGlobalKb={setUseGlobalKb}
                  onClose={() => setPickerOpen(false)}
                />
              )}

              {/* Main row: textarea + action buttons */}
              <div className="flex items-center gap-2 sm:gap-3">
                <textarea
                  ref={textareaRef}
                  className="flex-1 outline-none resize-none text-sm leading-6 bg-transparent placeholder-[#9CA3AF]"
                  placeholder={recording ? "Đang nghe… hãy nói câu hỏi của bạn" : "Ask me anything........"}
                  style={{ color: "#1A1A2E" }}
                  rows={1}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKey}
                  onPaste={handlePaste}
                  onFocus={() => setInputFocused(true)}
                />

                {/* Action buttons */}
                <div className="flex items-center gap-1 sm:gap-1.5 shrink-0">
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
                    title={recording ? "Dừng nhập bằng giọng nói" : "Nhập câu hỏi bằng giọng nói"}
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
                    className="h-9 px-3 sm:px-4 rounded-2xl flex items-center gap-2 text-white text-xs font-semibold transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
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
          {/* Drawer panel — mobile: full màn hình, tablet/desktop: panel bên phải */}
          <div
            className="fixed inset-y-0 right-0 z-[41] flex flex-col overflow-hidden w-full sm:w-[min(70vw,520px)] lg:w-[min(44vw,560px)]"
            style={{ animation: "slideInRight 0.28s cubic-bezier(0.4,0,0.2,1)" }}
          >
            <DocViewerPanel target={viewerTarget} onClose={() => setViewerTarget(null)} />
          </div>
        </>
      )}
    </div>
  );
}

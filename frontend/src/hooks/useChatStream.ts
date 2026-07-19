import { useCallback, useRef, useState } from "react";
import { chatStream } from "../api/client";
import type { Message, Source } from "../components/MessageBubble";

/** Body accepted by a single streamed exchange (mirrors chatStream's body). */
export interface StreamBody {
  question: string;
  history: object[];
  attachments?: string[];
  regenerate_message_id?: string;
  include_doc_ids?: string[];
  use_global_kb?: boolean;
}

/**
 * Owns the chat message list and the SSE streaming plumbing, extracted from
 * ChatPage (refactor E3). Callers keep the orchestration (uploads, session
 * creation, source-picker params) and just drive one exchange via `runStream`.
 */
export function useChatStream(onSessionCreated: (id: string) => void) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sending, setSending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patchMessage = useCallback((id: string, patch: Partial<Message>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  /** Drive one SSE exchange into the placeholder assistant message `aiId`. */
  const runStream = useCallback(async (
    sid: string,
    aiId: string,
    body: StreamBody,
  ) => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let content = "";
    // Tracks the message's live id: it may swap from the local placeholder
    // to the DB uuid on done, before the post-done suggestions arrive.
    let liveId = aiId;
    try {
      await chatStream(sid, body, {
        onMeta: (meta) => patchMessage(aiId, { sources: meta.sources as Source[] }),
        onStep: (step) => {
          setMessages((prev) => prev.map((m) =>
            m.id === aiId ? { ...m, steps: [...(m.steps ?? []), step] } : m
          ));
        },
        onDelta: (text) => {
          content += text;
          patchMessage(aiId, { content });
        },
        onDone: (done) => {
          // Swap in the DB id last so feedback targets the persisted row.
          if (done.message_id) liveId = done.message_id;
          patchMessage(aiId, { streaming: false, ...(done.message_id ? { id: done.message_id } : {}) });
          // First exchange sets an LLM-generated title — refresh the sidebar
          if (done.session_title) onSessionCreated(sid);
        },
        // Chips arrive after done; the message id may already be the DB uuid.
        onSuggestions: (questions) => patchMessage(liveId, { suggestions: questions }),
        onGrounding: (grounding) => patchMessage(liveId, { grounding }),
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
  }, [patchMessage, onSessionCreated]);

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return {
    messages, setMessages, sending, setSending,
    patchMessage, runStream, stopStreaming,
  };
}

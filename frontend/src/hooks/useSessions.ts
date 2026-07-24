import { useCallback, useEffect, useRef, useState } from "react";
import { sessionsAPI } from "../api/client";

export interface Session { id: string; title: string; created_at: string; }

const MAX_RETRIES = 20;
const RETRY_DELAY_MS = 2000;

/**
 * Owns the chat-session list: loads it (with retry), refetches on window
 * focus, and exposes `reload` for after create/delete. Lifted out of
 * SessionSidebar so the sidebar (list + search) and ChatPage (export uses
 * the active session's real title) read from one source of truth.
 *
 * `enabled` = người dùng đã đăng nhập. Khi chưa đăng nhập, KHÔNG được gọi
 * /api/sessions: request sẽ 401, interceptor ép reload về /login, và vì hook
 * này chạy vô điều kiện trong App (trước guard !user) nó sẽ lặp vô hạn ngay
 * trên trang đăng nhập. Khi `enabled` chuyển true (vừa đăng nhập) hook tự nạp.
 */
export function useSessions(enabled: boolean) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [status, setStatus] = useState<"loading" | "done" | "error">("loading");

  const timerRef = useRef<number | undefined>(undefined);
  const generationRef = useRef(0);

  const reload = useCallback((attempt = 0) => {
    const generation = ++generationRef.current;
    window.clearTimeout(timerRef.current);
    if (attempt === 0) setStatus("loading");
    sessionsAPI
      .list()
      .then((data) => {
        if (generation !== generationRef.current) return;
        setSessions(data);
        setStatus("done");
      })
      .catch(() => {
        if (generation !== generationRef.current) return;
        if (attempt < MAX_RETRIES) {
          timerRef.current = window.setTimeout(() => reload(attempt + 1), RETRY_DELAY_MS);
        } else {
          setStatus("error");
        }
      });
  }, []);

  useEffect(() => {
    // Chưa đăng nhập: dọn state, huỷ mọi retry đang chờ, KHÔNG gọi API.
    if (!enabled) {
      window.clearTimeout(timerRef.current);
      generationRef.current++;
      setSessions([]);
      setStatus("loading");
      return;
    }
    reload();
    const onFocus = () => reload();
    window.addEventListener("focus", onFocus);
    return () => {
      window.removeEventListener("focus", onFocus);
      window.clearTimeout(timerRef.current);
      generationRef.current++;
    };
  }, [enabled, reload]);

  return { sessions, status, reload };
}

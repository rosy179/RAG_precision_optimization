import { useCallback, useEffect, useRef, useState } from "react";
import {
  Trash2,
  Loader2,
  RefreshCw,
  Search,
  Edit,
  PanelLeftClose,
  Settings,
  LogOut,
} from "lucide-react";
import AiRobotIcon from "./AiRobotIcon";
import { sessionsAPI } from "../api/client";

interface Session { id: string; title: string; created_at: string; }

interface Props {
  user: { email: string; name?: string };
  onLogout: () => void;
  onNewChat: () => void;
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  refreshKey: number;
  isOpen: boolean;
  onToggle: () => void;
}

function groupByDate(sessions: Session[]) {
  const today = new Date().toDateString();
  const yesterday = new Date(Date.now() - 86400000).toDateString();
  const groups: Record<string, Session[]> = {};
  for (const s of sessions) {
    const d = new Date(s.created_at).toDateString();
    const label = d === today ? "Hôm nay" : d === yesterday ? "Hôm qua" : new Date(s.created_at).toLocaleDateString("vi-VN");
    if (!groups[label]) groups[label] = [];
    groups[label].push(s);
  }
  return groups;
}

const MAX_RETRIES = 20;
const RETRY_DELAY_MS = 2000;

export default function SessionSidebar({
  user,
  onLogout,
  onNewChat,
  activeId,
  onSelect,
  onDelete,
  refreshKey,
  isOpen,
  onToggle
}: Props) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [status, setStatus] = useState<"loading" | "done" | "error">("loading");
  const [profileOpen, setProfileOpen] = useState(false);

  const timerRef = useRef<number | undefined>(undefined);
  const generationRef = useRef(0);

  const load = useCallback((attempt = 0) => {
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
          timerRef.current = window.setTimeout(() => load(attempt + 1), RETRY_DELAY_MS);
        } else {
          setStatus("error");
        }
      });
  }, []);

  useEffect(() => {
    load();
    const onFocus = () => load();
    window.addEventListener("focus", onFocus);
    return () => {
      window.removeEventListener("focus", onFocus);
      window.clearTimeout(timerRef.current);
      generationRef.current++;
    };
  }, [refreshKey, load]);

  // Click outside profile popup to close
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('.profile-menu-container')) {
        setProfileOpen(false);
      }
    };
    if (profileOpen) {
      window.addEventListener('click', onClick);
    }
    return () => window.removeEventListener('click', onClick);
  }, [profileOpen]);

  const groups = groupByDate(sessions);
  const initial = user.name ? user.name[0].toUpperCase() : user.email[0].toUpperCase();

  return (
    <aside
      className="sidebar-panel flex flex-col h-full shrink-0 text-[#1A1A2E]"
      style={{
        width: isOpen ? 260 : 0,
        opacity: isOpen ? 1 : 0,
        background: "rgba(255, 255, 255, 0.25)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderRight: isOpen ? "1px solid rgba(255, 255, 255, 0.4)" : "none",
        overflow: "hidden",
        transition: "width 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.2s ease",
      }}
    >
      <div style={{ minWidth: 260 }} className="flex flex-col h-full">
        {/* Top Header */}
        <div className="flex items-center justify-between px-4 py-4">
          <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
            <AiRobotIcon mini />
            <span className="font-semibold text-lg" style={{ fontFamily: "Inter, sans-serif" }}>AI Agent</span>
          </div>
          <button
            onClick={onToggle}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-gray-500 hover:bg-black/5 hover:text-gray-800 transition-colors"
          >
            <PanelLeftClose className="w-5 h-5" />
          </button>
        </div>

        {/* Main Actions */}
        <div className="px-3 space-y-0.5 mt-2 font-medium shrink-0">
          <button onClick={onNewChat} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm hover:bg-black/5 transition-colors">
            <Edit className="w-4 h-4" />
            Đoạn chat mới
          </button>
          <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm hover:bg-black/5 transition-colors">
            <Search className="w-4 h-4" />
            Tìm kiếm đoạn chat
          </button>
        </div>

        {/* Recent Sessions Header (Fixed) */}
        <div className="px-6 mt-6 mb-2 shrink-0">
          <p className="text-sm font-semibold">Gần đây</p>
        </div>

        {/* Recent Sessions List (Scrollable) */}
        <div className="flex-1 overflow-y-auto px-3 pb-2">
          
          {status === "loading" && sessions.length === 0 && (
            <div className="flex items-center justify-center gap-2 text-xs text-gray-400 mt-6">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Đang tải...
            </div>
          )}
          {status === "error" && (
            <div className="flex flex-col items-center gap-2 mt-6">
              <p className="text-xs text-gray-400">Không thể tải lịch sử</p>
              <button onClick={() => load()} className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded hover:bg-black/5">
                <RefreshCw className="w-3 h-3" /> Thử lại
              </button>
            </div>
          )}

          {Object.entries(groups).map(([label, list]) => (
            <div key={label} className="mt-4 first:mt-0">
              <p className="text-xs font-semibold text-gray-400 px-3 mb-1">{label}</p>
              <div className="space-y-0.5">
                {list.map((s) => (
                  <div
                    key={s.id}
                    onClick={() => onSelect(s.id)}
                    className={`group flex items-center justify-between px-3 py-2 rounded-xl text-sm cursor-pointer transition-colors ${
                      activeId === s.id ? "bg-black/10 font-medium" : "hover:bg-black/5 text-gray-700"
                    }`}
                  >
                    <span className="truncate pr-2">{s.title}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); onDelete(s.id); }}
                      className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 rounded-md hover:bg-black/5 transition-all shrink-0"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer Profile */}
        <div className="relative mt-2 p-3 profile-container">
          {profileOpen && (
            <div
              className="absolute bottom-full left-3 right-3 mb-1 rounded-2xl shadow-xl border border-gray-100 overflow-hidden"
              style={{ background: "#fff", zIndex: 50 }}
            >
              <div className="p-4 border-b border-gray-100 bg-gray-50/50">
                <p className="font-semibold text-sm text-gray-900 truncate">{user.name || user.email.split('@')[0]}</p>
                <p className="text-xs text-gray-500 truncate mt-0.5">{user.email}</p>
              </div>
              <div className="p-1.5">
                <button className="w-full flex items-center gap-3 px-3 py-2 rounded-xl text-sm text-gray-700 hover:bg-gray-100 transition-colors">
                  <Settings className="w-4 h-4" />
                  Cài đặt
                </button>
                <button
                  onClick={onLogout}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-xl text-sm text-red-600 hover:bg-red-50 transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                  Đăng xuất
                </button>
              </div>
            </div>
          )}

          <div
            onClick={(e) => { e.stopPropagation(); setProfileOpen(!profileOpen); }}
            className="flex items-center gap-3 p-2 rounded-xl cursor-pointer hover:bg-black/5 transition-colors profile-menu-container"
          >
            <div
              className="w-9 h-9 rounded-full flex items-center justify-center text-white text-sm font-semibold shrink-0"
              style={{ background: "linear-gradient(135deg, #FF8CC8, #7C3AED)" }}
            >
              {initial}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold truncate">{user.name || user.email.split('@')[0]}</p>
            </div>
          </div>
        </div>

      </div>
    </aside>
  );
}

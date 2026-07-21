import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Trash2,
  Loader2,
  RefreshCw,
  Search,
  Edit,
  Activity,
  BookOpen,
  PanelLeftClose,
  Settings,
  LogOut,
  X,
} from "lucide-react";
import AiRobotIcon from "./AiRobotIcon";
import type { Session } from "../hooks/useSessions";

interface Props {
  user: { email: string; name?: string };
  isAdmin: boolean;
  onLogout: () => void;
  onNewChat: () => void;
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  sessions: Session[];
  sessionsStatus: "loading" | "done" | "error";
  onReload: () => void;
  isOpen: boolean;
  onToggle: () => void;
  /** Mobile: hiển thị dạng lớp phủ (drawer) thay vì chiếm chỗ layout */
  overlay?: boolean;
  knowledgeActive: boolean;
  onOpenKnowledge: () => void;
  dashboardActive: boolean;
  onOpenDashboard: () => void;
}

/** Lowercase + strip Vietnamese diacritics so search matches "cloud" against
 *  "Cloud WAF" and "dich vu" against "dịch vụ". */
function normalize(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d");
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

export default function SessionSidebar({
  user,
  isAdmin,
  onLogout,
  onNewChat,
  activeId,
  onSelect,
  onDelete,
  sessions,
  sessionsStatus,
  onReload,
  isOpen,
  onToggle,
  overlay = false,
  knowledgeActive,
  onOpenKnowledge,
  dashboardActive,
  onOpenDashboard
}: Props) {
  const [profileOpen, setProfileOpen] = useState(false);
  const [confirmTarget, setConfirmTarget] = useState<Session | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [search, setSearch] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Focus the search field the moment it opens
  useEffect(() => {
    if (searchOpen) searchInputRef.current?.focus();
  }, [searchOpen]);

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

  // Esc để đóng popup xác nhận xóa
  useEffect(() => {
    if (!confirmTarget) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setConfirmTarget(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [confirmTarget]);

  const filtered = useMemo(() => {
    const q = normalize(search.trim());
    if (!q) return sessions;
    return sessions.filter((s) => normalize(s.title).includes(q));
  }, [sessions, search]);

  const groups = groupByDate(filtered);
  const initial = user.name ? user.name[0].toUpperCase() : user.email[0].toUpperCase();

  const closeSearch = () => { setSearchOpen(false); setSearch(""); };

  return (
    <aside
      className={`sidebar-panel flex flex-col h-full shrink-0 text-[#1A1A2E] ${
        overlay ? "fixed inset-y-0 left-0 z-50" : ""
      }`}
      style={{
        width: isOpen ? 260 : 0,
        opacity: isOpen ? 1 : 0,
        background: overlay ? "rgba(255, 255, 255, 0.92)" : "rgba(255, 255, 255, 0.25)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderRight: isOpen ? "1px solid rgba(255, 255, 255, 0.4)" : "none",
        boxShadow: overlay && isOpen ? "8px 0 32px rgba(26,26,46,0.18)" : "none",
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
          {searchOpen ? (
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl"
              style={{ background: "rgba(0,0,0,0.04)" }}
            >
              <Search className="w-4 h-4 shrink-0 text-gray-500" />
              <input
                ref={searchInputRef}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Escape") closeSearch(); }}
                placeholder="Tìm theo tên đoạn chat…"
                className="flex-1 min-w-0 bg-transparent outline-none text-sm placeholder-gray-400"
              />
              <button
                onClick={closeSearch}
                title="Đóng tìm kiếm"
                className="shrink-0 p-0.5 text-gray-400 hover:text-gray-700 rounded transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setSearchOpen(true)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm hover:bg-black/5 transition-colors"
            >
              <Search className="w-4 h-4" />
              Tìm kiếm đoạn chat
            </button>
          )}
          <button
            onClick={onOpenKnowledge}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-colors ${
              knowledgeActive ? "bg-black/10 font-semibold" : "hover:bg-black/5"
            }`}
          >
            <BookOpen className="w-4 h-4" style={{ color: "#3B82F6" }} />
            Kho kiến thức
          </button>
          {isAdmin && (
            <button
              onClick={onOpenDashboard}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-colors ${
                dashboardActive ? "bg-black/10 font-semibold" : "hover:bg-black/5"
              }`}
            >
              <Activity className="w-4 h-4" style={{ color: "#7C3AED" }} />
              Thống kê
            </button>
          )}
        </div>

        {/* Recent Sessions Header (Fixed) */}
        <div className="px-6 mt-6 mb-2 shrink-0">
          <p className="text-sm font-semibold">Gần đây</p>
        </div>

        {/* Recent Sessions List (Scrollable) */}
        <div className="flex-1 overflow-y-auto px-3 pb-2">

          {sessionsStatus === "loading" && sessions.length === 0 && (
            <div className="flex items-center justify-center gap-2 text-xs text-gray-400 mt-6">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Đang tải...
            </div>
          )}
          {sessionsStatus === "error" && (
            <div className="flex flex-col items-center gap-2 mt-6">
              <p className="text-xs text-gray-400">Không thể tải lịch sử</p>
              <button onClick={() => onReload()} className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded hover:bg-black/5">
                <RefreshCw className="w-3 h-3" /> Thử lại
              </button>
            </div>
          )}
          {sessionsStatus === "done" && search.trim() && filtered.length === 0 && (
            <p className="text-xs text-gray-400 text-center mt-6">
              Không tìm thấy đoạn chat nào khớp “{search.trim()}”.
            </p>
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
                      onClick={(e) => { e.stopPropagation(); setConfirmTarget(s); }}
                      title="Xóa cuộc hội thoại"
                      className="opacity-100 md:opacity-0 md:group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 rounded-md hover:bg-black/5 transition-all shrink-0"
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

      {/* Popup xác nhận xóa — portal ra body để không bị sidebar (overflow/backdrop-filter) cắt mất */}
      {confirmTarget &&
        createPortal(
          <div
            className="fixed inset-0 z-[100] flex items-center justify-center p-4"
            style={{ background: "rgba(26, 26, 46, 0.45)" }}
            onClick={() => setConfirmTarget(null)}
          >
            <div
              role="dialog"
              aria-modal="true"
              aria-labelledby="delete-session-title"
              className="w-full max-w-sm rounded-2xl bg-white shadow-2xl p-5"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-full flex items-center justify-center shrink-0 bg-red-50">
                  <Trash2 className="w-5 h-5 text-red-500" />
                </div>
                <div className="min-w-0">
                  <p id="delete-session-title" className="font-semibold text-[15px] text-gray-900">
                    Xóa cuộc hội thoại?
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    "<span className="font-medium text-gray-700">{confirmTarget.title}</span>" sẽ bị
                    xóa vĩnh viễn và không thể hoàn tác.
                  </p>
                </div>
              </div>
              <div className="flex justify-end gap-2 mt-5">
                <button
                  onClick={() => setConfirmTarget(null)}
                  autoFocus
                  className="px-4 py-2 rounded-xl text-sm font-medium text-gray-600 hover:bg-black/5 transition-colors"
                >
                  Hủy
                </button>
                <button
                  onClick={() => {
                    onDelete(confirmTarget.id);
                    setConfirmTarget(null);
                  }}
                  className="px-4 py-2 rounded-xl text-sm font-semibold text-white transition-opacity hover:opacity-90"
                  style={{ background: "#EF4444" }}
                >
                  Xóa
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}
    </aside>
  );
}

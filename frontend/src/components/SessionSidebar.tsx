import { useEffect, useState } from "react";
import { MessageSquare, Trash2, ChevronLeft } from "lucide-react";
import { sessionsAPI } from "../api/client";

interface Session { id: string; title: string; created_at: string; }

interface Props {
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  refreshKey: number;
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

export default function SessionSidebar({ activeId, onSelect, onDelete, refreshKey }: Props) {
  const [sessions, setSessions] = useState<Session[]>([]);

  useEffect(() => {
    sessionsAPI.list().then(setSessions).catch(() => {});
  }, [refreshKey]);

  const groups = groupByDate(sessions);

  return (
    <aside className="flex flex-col bg-[#F0EEFF] w-[280px] min-h-screen shrink-0 border-r border-[#E0D9FF]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-5 pb-3">
        <h2 className="font-semibold text-[#1A1A2E] text-base">Chat results</h2>
        <button className="w-8 h-8 rounded-lg flex items-center justify-center text-gray-400 hover:bg-white hover:text-[#7C3AED] transition-all">
          <ChevronLeft className="w-4 h-4" />
        </button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-4">
        {Object.keys(groups).length === 0 && (
          <p className="text-xs text-gray-400 text-center mt-8">Chưa có cuộc trò chuyện nào</p>
        )}
        {Object.entries(groups).map(([label, list]) => (
          <div key={label}>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-2 mb-2">{label}</p>
            <div className="space-y-1">
              {list.map((s) => (
                <div
                  key={s.id}
                  onClick={() => onSelect(s.id)}
                  className={`session-card group ${activeId === s.id ? "active" : ""}`}
                >
                  <div className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0 bg-[#EDE9FE]">
                    <MessageSquare className="w-4 h-4 text-[#7C3AED]" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-[#1A1A2E] truncate">{s.title}</p>
                    <p className="text-xs text-gray-400">
                      {new Date(s.created_at).toLocaleString("vi-VN", { hour: "2-digit", minute: "2-digit" })}
                    </p>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDelete(s.id); }}
                    className="opacity-0 group-hover:opacity-100 w-6 h-6 rounded-lg flex items-center justify-center text-gray-400 hover:text-red-500 hover:bg-red-50 transition-all"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}

import { MessageSquare, Plus, Star, FolderOpen, Settings, LogOut } from "lucide-react";
import AiRobotIcon from "./AiRobotIcon";

interface Props {
  onNewChat: () => void;
  onLogout: () => void;
  userEmail: string;
  onToggleSidebar: () => void;
  sidebarOpen: boolean;
}

export default function IconRail({ onNewChat, onLogout, userEmail, onToggleSidebar, sidebarOpen }: Props) {
  const initial = userEmail?.[0]?.toUpperCase() || "U";

  return (
    <aside
      className="flex flex-col items-center py-4 gap-3 w-[64px] min-h-screen shrink-0"
      style={{
        background: "rgba(255,255,255,0.45)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        borderRight: "1px solid rgba(255,255,255,0.65)",
        boxShadow: "2px 0 16px rgba(124,58,237,0.08)",
      }}
    >
      {/* Logo — mini robot icon */}
      <div className="mb-2 cursor-pointer">
        <AiRobotIcon mini />
      </div>

      {/* New Chat */}
      <button
        onClick={onNewChat}
        title="New Chat"
        className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200"
        style={{
          color: "#7C3AED",
          background: "rgba(124,58,237,0.08)",
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.background = "rgba(124,58,237,0.18)";
          (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 14px rgba(124,58,237,0.25)";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.background = "rgba(124,58,237,0.08)";
          (e.currentTarget as HTMLElement).style.boxShadow = "";
        }}
      >
        <Plus className="w-5 h-5" />
      </button>

      {/* Chat (toggle sidebar) */}
      <button
        onClick={onToggleSidebar}
        title="Chats"
        className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300"
        style={{
          background: sidebarOpen
            ? "linear-gradient(135deg, #7C3AED, #3B82F6)"
            : "rgba(124,58,237,0.08)",
          color: sidebarOpen ? "#fff" : "#7C3AED",
          boxShadow: sidebarOpen ? "0 4px 14px rgba(124,58,237,0.45)" : "none",
        }}
      >
        <MessageSquare className="w-5 h-5" />
      </button>

      {/* Starred */}
      <button
        title="Yêu thích"
        className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200"
        style={{ color: "#94A3B8" }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.background = "rgba(124,58,237,0.1)";
          (e.currentTarget as HTMLElement).style.color = "#7C3AED";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.background = "";
          (e.currentTarget as HTMLElement).style.color = "#94A3B8";
        }}
      >
        <Star className="w-5 h-5" />
      </button>

      {/* Documents */}
      <button
        title="Tài liệu"
        className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200"
        style={{ color: "#94A3B8" }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.background = "rgba(124,58,237,0.1)";
          (e.currentTarget as HTMLElement).style.color = "#7C3AED";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.background = "";
          (e.currentTarget as HTMLElement).style.color = "#94A3B8";
        }}
      >
        <FolderOpen className="w-5 h-5" />
      </button>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Settings */}
      <button
        title="Cài đặt"
        className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200"
        style={{ color: "#94A3B8" }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.background = "rgba(124,58,237,0.1)";
          (e.currentTarget as HTMLElement).style.color = "#7C3AED";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.background = "";
          (e.currentTarget as HTMLElement).style.color = "#94A3B8";
        }}
      >
        <Settings className="w-5 h-5" />
      </button>

      {/* Logout */}
      <button
        onClick={onLogout}
        title="Đăng xuất"
        className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200"
        style={{ color: "#94A3B8" }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.background = "rgba(239,68,68,0.1)";
          (e.currentTarget as HTMLElement).style.color = "#EF4444";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.background = "";
          (e.currentTarget as HTMLElement).style.color = "#94A3B8";
        }}
      >
        <LogOut className="w-4 h-4" />
      </button>

      {/* Avatar */}
      <div
        className="w-10 h-10 rounded-full flex items-center justify-center text-white text-sm font-semibold mt-1 shadow-lg"
        style={{
          background: "linear-gradient(135deg, #7C3AED, #3B82F6)",
          boxShadow: "0 4px 12px rgba(124,58,237,0.4)",
        }}
      >
        {initial}
      </div>
    </aside>
  );
}

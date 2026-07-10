import { MessageSquare, Plus, Star, FolderOpen, Settings, LogOut, Bot } from "lucide-react";

interface Props {
  onNewChat: () => void;
  onLogout: () => void;
  userEmail: string;
}

export default function IconRail({ onNewChat, onLogout, userEmail }: Props) {
  const initial = userEmail?.[0]?.toUpperCase() || "U";

  return (
    <aside className="flex flex-col items-center py-4 gap-3 bg-[#1A1A2E] w-[64px] min-h-screen shrink-0">
      {/* Logo */}
      <div className="w-10 h-10 rounded-xl bg-[#7C3AED] flex items-center justify-center mb-2 shadow-lg">
        <Bot className="w-5 h-5 text-white" />
      </div>

      {/* New Chat */}
      <button
        onClick={onNewChat}
        title="New Chat"
        className="w-10 h-10 rounded-xl flex items-center justify-center text-[#94A3B8] hover:bg-[#252547] hover:text-white transition-all"
      >
        <Plus className="w-5 h-5" />
      </button>

      {/* Chat (active) */}
      <button
        title="Chats"
        className="w-10 h-10 rounded-xl flex items-center justify-center bg-[#7C3AED] text-white shadow"
      >
        <MessageSquare className="w-5 h-5" />
      </button>

      {/* Starred */}
      <button
        title="Yêu thích"
        className="w-10 h-10 rounded-xl flex items-center justify-center text-[#94A3B8] hover:bg-[#252547] hover:text-white transition-all"
      >
        <Star className="w-5 h-5" />
      </button>

      {/* Documents */}
      <button
        title="Tài liệu"
        className="w-10 h-10 rounded-xl flex items-center justify-center text-[#94A3B8] hover:bg-[#252547] hover:text-white transition-all"
      >
        <FolderOpen className="w-5 h-5" />
      </button>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Settings */}
      <button
        title="Cài đặt"
        className="w-10 h-10 rounded-xl flex items-center justify-center text-[#94A3B8] hover:bg-[#252547] hover:text-white transition-all"
      >
        <Settings className="w-5 h-5" />
      </button>

      {/* Logout */}
      <button
        onClick={onLogout}
        title="Đăng xuất"
        className="w-10 h-10 rounded-xl flex items-center justify-center text-[#94A3B8] hover:bg-red-900/30 hover:text-red-400 transition-all"
      >
        <LogOut className="w-4 h-4" />
      </button>

      {/* Avatar */}
      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#7C3AED] to-[#3B82F6] flex items-center justify-center text-white text-sm font-semibold mt-1">
        {initial}
      </div>
    </aside>
  );
}

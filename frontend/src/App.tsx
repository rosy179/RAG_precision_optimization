import { useState, useCallback } from "react";
import { useAuth } from "./hooks/useAuth";
import AuthPage from "./pages/AuthPage";
import ChatPage from "./pages/ChatPage";
import IconRail from "./components/IconRail";
import SessionSidebar from "./components/SessionSidebar";
import { sessionsAPI } from "./api/client";

export default function App() {
  const { user, login, register, logout } = useAuth();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sidebarKey, setSidebarKey] = useState(0);

  const handleAuth = useCallback(async (
    email: string, password: string, name?: string, isRegister?: boolean
  ) => {
    if (isRegister) await register(email, password, name || "");
    else await login(email, password);
  }, [login, register]);

  const handleNewChat = useCallback(async () => {
    setActiveSessionId(null);
  }, []);

  const handleDeleteSession = useCallback(async (id: string) => {
    await sessionsAPI.delete(id).catch(() => {});
    if (activeSessionId === id) setActiveSessionId(null);
    setSidebarKey((k) => k + 1);
  }, [activeSessionId]);

  const handleSessionCreated = useCallback((id: string) => {
    setActiveSessionId(id);
    setSidebarKey((k) => k + 1);
  }, []);

  if (!user) {
    return <AuthPage onAuth={handleAuth} />;
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#F0EEFF]">
      {/* 1. Narrow icon rail */}
      <IconRail
        onNewChat={handleNewChat}
        onLogout={logout}
        userEmail={user.email}
      />

      {/* 2. Session sidebar */}
      <SessionSidebar
        activeId={activeSessionId}
        onSelect={setActiveSessionId}
        onDelete={handleDeleteSession}
        refreshKey={sidebarKey}
      />

      {/* 3. Chat area */}
      <div className="flex-1 min-w-0 rounded-tl-3xl rounded-bl-3xl overflow-hidden shadow-xl">
        <ChatPage
          sessionId={activeSessionId}
          onSessionCreated={handleSessionCreated}
        />
      </div>
    </div>
  );
}

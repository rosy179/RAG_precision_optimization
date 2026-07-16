import { useState, useCallback } from "react";
import { PanelLeftOpen } from "lucide-react";
import { useAuth } from "./hooks/useAuth";
import AuthPage from "./pages/AuthPage";
import ChatPage from "./pages/ChatPage";
import KnowledgePage from "./pages/KnowledgePage";
import DashboardPage from "./pages/DashboardPage";
import SessionSidebar from "./components/SessionSidebar";
import { sessionsAPI } from "./api/client";

export default function App() {
  const { user, login, register, logout } = useAuth();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sidebarKey, setSidebarKey] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [view, setView] = useState<"chat" | "knowledge" | "dashboard">("chat");

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  const handleAuth = useCallback(async (
    email: string, password: string, name?: string, isRegister?: boolean
  ) => {
    if (isRegister) await register(email, password, name || "");
    else await login(email, password);
  }, [login, register]);

  const handleNewChat = useCallback(async () => {
    setActiveSessionId(null);
    setView("chat");
  }, []);

  const handleSelectSession = useCallback((id: string) => {
    setActiveSessionId(id);
    setView("chat");
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
    <div className="relative flex h-screen w-screen overflow-hidden bg-[#F4F7FE]">
      {/* Abstract Background Blobs */}
      <div className="absolute top-[-10%] left-[-10%] w-[45%] h-[45%] rounded-full filter blur-[100px] opacity-80" style={{ background: "linear-gradient(135deg, #A8C8FF, #D9E4FF)" }} />
      <div className="absolute top-[5%] right-[-5%] w-[50%] h-[50%] rounded-full filter blur-[120px] opacity-70" style={{ background: "linear-gradient(135deg, #FFD1ED, #FFE4F4)" }} />
      <div className="absolute bottom-[-10%] left-[15%] w-[60%] h-[60%] rounded-full filter blur-[130px] opacity-70" style={{ background: "linear-gradient(135deg, #E0C3FC, #8EC5FC)" }} />
      
      {/* Decorative top-left ring */}
      <div className="absolute top-10 left-10 w-[60px] h-[60px] rounded-full border-[5px] border-white/90 flex items-center justify-center z-0 backdrop-blur-sm pointer-events-none" style={{ boxShadow: '0 0 25px rgba(255,255,255,1), inset 0 0 15px rgba(255,255,255,0.8)' }}>
        <div className="w-2.5 h-2.5 rounded-full bg-white shadow-[0_0_8px_white] absolute top-2 left-2" />
      </div>

      <div className="flex h-full w-full z-10 relative">
        {/* Session sidebar */}
        <SessionSidebar
        user={user}
        onLogout={logout}
        onNewChat={handleNewChat}
        activeId={view === "chat" ? activeSessionId : null}
        onSelect={handleSelectSession}
        onDelete={handleDeleteSession}
        refreshKey={sidebarKey}
        isOpen={sidebarOpen}
        onToggle={toggleSidebar}
        knowledgeActive={view === "knowledge"}
        onOpenKnowledge={() => setView("knowledge")}
        dashboardActive={view === "dashboard"}
        onOpenDashboard={() => setView("dashboard")}
      />

      {/* Chat area */}
      <div className="flex-1 min-w-0 overflow-hidden relative">
        {!sidebarOpen && (
          <button
            onClick={toggleSidebar}
            className="absolute top-4 left-4 z-50 w-10 h-10 rounded-xl flex items-center justify-center bg-white/80 backdrop-blur-md shadow-sm border border-gray-200 text-gray-500 hover:text-gray-800 transition-all hover:bg-white"
          >
            <PanelLeftOpen className="w-5 h-5" />
          </button>
        )}
        {view === "knowledge" ? (
          <KnowledgePage />
        ) : view === "dashboard" ? (
          <DashboardPage />
        ) : (
          <ChatPage
            sessionId={activeSessionId}
            onSessionCreated={handleSessionCreated}
          />
        )}
      </div>
      </div>
    </div>
  );
}

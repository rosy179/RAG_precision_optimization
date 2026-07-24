import { useState, useCallback, useEffect } from "react";
import { PanelLeftOpen } from "lucide-react";
import { useAuth } from "./hooks/useAuth";
import { useIsMobile } from "./hooks/useIsMobile";
import { useSessions } from "./hooks/useSessions";
import { useI18n } from "./i18n/LanguageProvider";
import AuthPage from "./pages/AuthPage";
import ChatPage from "./pages/ChatPage";
import KnowledgePage from "./pages/KnowledgePage";
import DashboardPage from "./pages/DashboardPage";
import SessionSidebar from "./components/SessionSidebar";
import ErrorBoundary from "./components/ErrorBoundary";
import { sessionsAPI } from "./api/client";

export default function App() {
  const { user, login, register, logout } = useAuth();
  const { t } = useI18n();
  const isMobile = useIsMobile();
  const { sessions, status: sessionsStatus, reload: reloadSessions } = useSessions(!!user);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(
    () => window.matchMedia("(min-width: 768px)").matches
  );
  const [view, setView] = useState<"chat" | "knowledge" | "dashboard">("chat");
  const isAdmin = !!user?.is_admin;
  const activeSession = sessions.find((s) => s.id === activeSessionId);

  // Đổi hướng / đổi cỡ màn hình: mobile mặc định đóng, desktop mặc định mở
  useEffect(() => {
    setSidebarOpen(!isMobile);
  }, [isMobile]);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  // Trên mobile sidebar là lớp phủ — chọn xong thì tự đóng
  const closeSidebarOnMobile = useCallback(() => {
    if (window.matchMedia("(max-width: 767px)").matches) setSidebarOpen(false);
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
    closeSidebarOnMobile();
  }, [closeSidebarOnMobile]);

  const handleSelectSession = useCallback((id: string) => {
    setActiveSessionId(id);
    setView("chat");
    closeSidebarOnMobile();
  }, [closeSidebarOnMobile]);

  const handleDeleteSession = useCallback(async (id: string) => {
    await sessionsAPI.delete(id).catch(() => {});
    if (activeSessionId === id) setActiveSessionId(null);
    reloadSessions();
  }, [activeSessionId, reloadSessions]);

  const handleSessionCreated = useCallback((id: string) => {
    setActiveSessionId(id);
    reloadSessions();
  }, [reloadSessions]);

  // Non-admins can't reach the dashboard (its data is admin-only); if a stale
  // view lands there after permissions change, fall back to chat.
  useEffect(() => {
    if (view === "dashboard" && !isAdmin) setView("chat");
  }, [view, isAdmin]);

  if (!user) {
    return <AuthPage onAuth={handleAuth} />;
  }

  return (
    <div className="relative flex h-dvh w-full overflow-hidden bg-[#F4F7FE]">
      {/* Abstract Background Blobs */}
      <div className="absolute top-[-10%] left-[-10%] w-[45%] h-[45%] rounded-full filter blur-[100px] opacity-80" style={{ background: "linear-gradient(135deg, #A8C8FF, #D9E4FF)" }} />
      <div className="absolute top-[5%] right-[-5%] w-[50%] h-[50%] rounded-full filter blur-[120px] opacity-70" style={{ background: "linear-gradient(135deg, #FFD1ED, #FFE4F4)" }} />
      <div className="absolute bottom-[-10%] left-[15%] w-[60%] h-[60%] rounded-full filter blur-[130px] opacity-70" style={{ background: "linear-gradient(135deg, #E0C3FC, #8EC5FC)" }} />

      {/* Decorative top-left ring (desktop only) */}
      <div className="hidden md:flex absolute top-10 left-10 w-[60px] h-[60px] rounded-full border-[5px] border-white/90 items-center justify-center z-0 backdrop-blur-sm pointer-events-none" style={{ boxShadow: '0 0 25px rgba(255,255,255,1), inset 0 0 15px rgba(255,255,255,0.8)' }}>
        <div className="w-2.5 h-2.5 rounded-full bg-white shadow-[0_0_8px_white] absolute top-2 left-2" />
      </div>

      <div className="flex h-full w-full z-10 relative">
        {/* Mobile: backdrop đóng sidebar khi chạm ra ngoài */}
        {isMobile && sidebarOpen && (
          <div
            onClick={toggleSidebar}
            className="fixed inset-0 z-40 bg-black/25 backdrop-blur-[2px]"
            style={{ animation: "fadeIn 0.2s ease" }}
          />
        )}

        {/* Session sidebar */}
        <SessionSidebar
        user={user}
        isAdmin={isAdmin}
        onLogout={logout}
        onNewChat={handleNewChat}
        activeId={view === "chat" ? activeSessionId : null}
        onSelect={handleSelectSession}
        onDelete={handleDeleteSession}
        sessions={sessions}
        sessionsStatus={sessionsStatus}
        onReload={reloadSessions}
        isOpen={sidebarOpen}
        onToggle={toggleSidebar}
        overlay={isMobile}
        knowledgeActive={view === "knowledge"}
        onOpenKnowledge={() => { setView("knowledge"); closeSidebarOnMobile(); }}
        dashboardActive={view === "dashboard"}
        onOpenDashboard={() => { setView("dashboard"); closeSidebarOnMobile(); }}
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
        {/* Crash của một trang chỉ thay trang đó bằng thông báo lỗi,
            sidebar vẫn dùng được; key={view} reset boundary khi chuyển trang */}
        <ErrorBoundary
          key={view}
          label={t("error.pageLabel")}
          title={t("error.title")}
          descTemplate={t("error.desc")}
          retryLabel={t("common.retry")}
          reloadLabel={t("error.reload")}
        >
          {view === "knowledge" ? (
            <KnowledgePage />
          ) : view === "dashboard" && isAdmin ? (
            <DashboardPage />
          ) : (
            <ChatPage
              sessionId={activeSessionId}
              sessionTitle={activeSession?.title ?? ""}
              onSessionCreated={handleSessionCreated}
            />
          )}
        </ErrorBoundary>
      </div>
      </div>
    </div>
  );
}

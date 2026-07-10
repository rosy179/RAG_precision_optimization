import { useState, type FormEvent } from "react";
import { Bot, Loader2 } from "lucide-react";

interface Props { onAuth: (email: string, password: string, name?: string, isRegister?: boolean) => Promise<void>; }

export default function AuthPage({ onAuth }: Props) {
  const [tab, setTab] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await onAuth(email, password, name, tab === "register");
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Đã có lỗi xảy ra");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-gradient-to-br from-[#F0EEFF] to-[#E0D9FF]">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-[#7C3AED] flex items-center justify-center shadow-lg mb-4">
            <Bot className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-[#1A1A2E]">IT Knowledge Assistant</h1>
          <p className="text-sm text-gray-500 mt-1">RAG-powered chatbot cho IT / AI / Security</p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-3xl shadow-card p-8">
          {/* Tabs */}
          <div className="flex bg-[#F0EEFF] rounded-2xl p-1 mb-6">
            {(["login", "register"] as const).map((t) => (
              <button
                key={t}
                onClick={() => { setTab(t); setError(""); }}
                className={`flex-1 py-2 text-sm font-medium rounded-xl transition-all ${
                  tab === t ? "bg-white text-[#7C3AED] shadow-sm" : "text-gray-500 hover:text-[#7C3AED]"
                }`}
              >
                {t === "login" ? "Đăng nhập" : "Đăng ký"}
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="space-y-4">
            {tab === "register" && (
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1.5">Tên của bạn</label>
                <input
                  className="input-base"
                  placeholder="Nguyễn Văn A"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
            )}
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1.5">Email</label>
              <input
                type="email"
                className="input-base"
                placeholder="you@company.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1.5">Mật khẩu</label>
              <input
                type="password"
                className="input-base"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
              />
            </div>

            {error && (
              <p className="text-red-500 text-xs bg-red-50 rounded-xl px-3 py-2">{error}</p>
            )}

            <button type="submit" disabled={loading} className="btn-primary w-full mt-2 flex items-center justify-center gap-2">
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {tab === "login" ? "Đăng nhập" : "Tạo tài khoản"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

import { useState, type FormEvent } from "react";
import { Loader2, Eye, EyeOff } from "lucide-react";
import AiRobotIcon from "../components/AiRobotIcon";

interface Props { onAuth: (email: string, password: string, name?: string, isRegister?: boolean) => Promise<void>; }

export default function AuthPage({ onAuth }: Props) {
  const [tab, setTab] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

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
    <div
      className="flex h-screen w-screen items-center justify-center relative overflow-hidden"
      style={{
        background: "linear-gradient(135deg, #A8C8FF 0%, #FFD1ED 50%, #E0C3FC 100%)",
      }}
    >
      {/* Background blobs for depth */}
      <div className="absolute top-[-10%] left-[-10%] w-[60%] h-[60%] rounded-full bg-blue-300/40 blur-[120px]" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] rounded-full bg-pink-300/40 blur-[120px]" />

      {/* Main Container */}
      <div
        className="w-[92%] max-w-5xl h-[85vh] max-h-[750px] rounded-[40px] relative overflow-hidden flex shadow-2xl"
        style={{
          background: "rgba(255, 255, 255, 0.8)",
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          border: "1px solid rgba(255,255,255,0.9)",
        }}
      >
        {/* Top-left: Website description and AI Agent name */}
        <div className="absolute top-10 left-10 z-30">
          <h1 className="text-2xl font-bold text-[#1A1A2E] tracking-tight">AI Agent</h1>
          <p className="text-sm mt-1 font-medium" style={{ color: "#7C3AED" }}>IT Knowledge Assistant • RAG-powered</p>
        </div>

        {/* Left side: AI Robot decorative element replacing the bubbles */}
        <div className="hidden md:flex w-1/2 relative h-full items-center justify-center overflow-visible z-10 pl-8">
          
          {/* Mascot wrapper */}
          <div className="relative w-[400px] h-[400px] flex items-center justify-center overflow-visible z-20">
             
             {/* The tilted robot face + hands as one tilted unit */}
             <div className="transform rotate-[-20deg] scale-110 relative z-20">
               <AiRobotIcon size={340} />

               {/* Left hand bubble (pink/magenta) - overlaps lower-left of aura */}
               <div
                 className="absolute w-[100px] h-[100px] rounded-full bg-gradient-to-tr from-[#E040FB] to-[#F472B6] blur-[18px] opacity-90 animate-bounce mix-blend-screen"
                 style={{ bottom: "70px", left: "10px", animationDuration: "4s" }}
               />

               {/* Right hand bubble (blue/cyan) - overlaps lower-right of aura */}
               <div
                 className="absolute w-[100px] h-[100px] rounded-full bg-gradient-to-tr from-[#2979FF] to-[#67E8F9] blur-[18px] opacity-90 animate-bounce mix-blend-screen"
                 style={{ bottom: "70px", right: "10px", animationDuration: "5s", animationDelay: "1s" }}
               />
             </div>
             
          </div>

          {/* Floating decorative elements mimicking balls */}
          <div className="absolute top-[10%] left-[20%] w-16 h-16 rounded-full bg-gradient-to-tr from-cyan-300 to-blue-200 shadow-xl z-30" />
          <div className="absolute bottom-[25%] left-[5%] w-12 h-12 rounded-full bg-gradient-to-tr from-yellow-300 to-amber-200 shadow-xl z-30" />
          <div className="absolute top-[25%] right-[-5%] w-12 h-12 rounded-full bg-gradient-to-tr from-pink-300 to-rose-200 shadow-lg z-10" />
          <div className="absolute bottom-[20%] right-[10%] w-14 h-14 rounded-full bg-gradient-to-tr from-pink-200 to-purple-200 shadow-xl opacity-90 backdrop-blur-md z-30" />
        </div>

        {/* Right side: Login Form */}
        <div className="w-full md:w-1/2 flex flex-col justify-center items-center p-8 z-20 bg-white/30 backdrop-blur-sm">
          <div className="w-full max-w-sm">
            
            <div className="text-center mb-10">
              <h2 className="text-3xl font-semibold text-[#374151] mb-2" style={{ fontFamily: "Inter, sans-serif" }}>
                {tab === "login" ? "Log in" : "Sign up"}
              </h2>
            </div>

            <form onSubmit={submit} className="space-y-5">
              {tab === "register" && (
                <div>
                  <label className="block text-xs font-semibold mb-2 ml-2" style={{ color: "#6B7280" }}>Name</label>
                  <input
                    className="w-full bg-transparent border-2 border-[#E5E7EB] rounded-full px-5 py-3 text-sm outline-none focus:border-[#7C3AED] transition-colors"
                    placeholder="Nguyễn Văn A"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </div>
              )}
              <div>
                <label className="block text-xs font-semibold mb-2 ml-2" style={{ color: "#6B7280" }}>Login, email or phone number</label>
                <input
                  type="email"
                  className="w-full bg-transparent border-2 border-[#E5E7EB] rounded-full px-5 py-3 text-sm outline-none focus:border-[#7C3AED] transition-colors"
                  placeholder="you@company.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-semibold mb-2 ml-2" style={{ color: "#6B7280" }}>Password</label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    className="w-full bg-transparent border-2 border-[#E5E7EB] rounded-full px-5 py-3 text-sm outline-none focus:border-[#7C3AED] transition-colors"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    minLength={6}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {error && (
                <p className="text-xs text-red-500 text-center bg-red-50 rounded-full py-2 border border-red-100">
                  {error}
                </p>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full mt-2 text-white font-medium rounded-full px-5 py-3.5 shadow-md transition-all flex items-center justify-center gap-2"
                style={{
                  background: "linear-gradient(to right, #3B82F6, #2563EB)",
                  boxShadow: "0 4px 14px rgba(37, 99, 235, 0.4)"
                }}
                onMouseEnter={e => (e.currentTarget.style.background = "linear-gradient(to right, #2563EB, #1D4ED8)")}
                onMouseLeave={e => (e.currentTarget.style.background = "linear-gradient(to right, #3B82F6, #2563EB)")}
              >
                {loading && <Loader2 className="w-4 h-4 animate-spin" />}
                {tab === "login" ? "Log in" : "Create account"}
              </button>
            </form>

            <div className="mt-8 relative flex items-center justify-center">
              <div className="absolute w-full border-t border-gray-300"></div>
              <span className="px-4 text-xs font-medium text-gray-500 relative bg-transparent backdrop-blur-[2px]">
                or {tab === "login" ? "log in" : "sign up"} with
              </span>
            </div>

            <div className="flex justify-center gap-4 mt-6">
              <button className="w-[60px] h-[40px] rounded-xl border border-gray-300 flex items-center justify-center hover:bg-white transition-colors bg-white/50 backdrop-blur-sm">
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
              </button>
              <button className="w-[60px] h-[40px] rounded-xl border border-gray-300 flex items-center justify-center hover:bg-white transition-colors bg-white/50 backdrop-blur-sm">
                <svg className="w-5 h-5" viewBox="0 0 23 23">
                  <path fill="#f35325" d="M1 1h10v10H1z"/>
                  <path fill="#81bc06" d="M12 1h10v10H12z"/>
                  <path fill="#05a6f0" d="M1 12h10v10H1z"/>
                  <path fill="#ffba08" d="M12 12h10v10H12z"/>
                </svg>
              </button>
            </div>

            <div className="mt-8 text-center text-xs font-semibold" style={{ color: "#6B7280" }}>
              {tab === "login" && (
                <button className="hover:text-gray-900 transition-colors mb-2 block w-full">
                  Forgot login or password?
                </button>
              )}
              <button onClick={() => { setTab(tab === "login" ? "register" : "login"); setError(""); }} className="hover:text-[#7C3AED] transition-colors mt-2">
                {tab === "login" ? "Don't have an account? Sign up" : "Already have an account? Log in"}
              </button>
            </div>
            
          </div>
        </div>
      </div>
    </div>
  );
}

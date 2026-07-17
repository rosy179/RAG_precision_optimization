import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw, RotateCcw } from "lucide-react";

interface Props {
  children: ReactNode;
  /** Nhãn khu vực được bọc — hiển thị trong thông báo lỗi (vd. "trang này") */
  label?: string;
}

interface State {
  error: Error | null;
}

/** Chặn lỗi render của cây con: thay vì trắng cả app, hiển thị thông báo
 *  thân thiện kèm nút thử lại. React yêu cầu error boundary là class component. */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  private reset = () => this.setState({ error: null });

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <div className="flex flex-1 h-full w-full items-center justify-center p-6">
        <div
          className="w-full max-w-sm text-center rounded-3xl px-6 py-8"
          style={{
            background: "rgba(255,255,255,0.8)",
            backdropFilter: "blur(16px)",
            WebkitBackdropFilter: "blur(16px)",
            border: "1px solid rgba(239,68,68,0.18)",
            boxShadow: "0 8px 32px rgba(124,58,237,0.12)",
          }}
        >
          <div
            className="w-12 h-12 rounded-2xl mx-auto mb-4 flex items-center justify-center"
            style={{ background: "rgba(239,68,68,0.1)", color: "#EF4444" }}
          >
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h2 className="text-base font-bold text-[#1A1A2E]">Đã có lỗi xảy ra</h2>
          <p className="text-xs text-gray-500 mt-1.5">
            {this.props.label || "Phần này"} gặp sự cố khi hiển thị. Bạn có thể thử
            lại hoặc tải lại trang.
          </p>
          <p
            className="text-[11px] font-mono mt-3 px-3 py-2 rounded-xl break-words"
            style={{
              background: "rgba(239,68,68,0.06)",
              border: "1px solid rgba(239,68,68,0.12)",
              color: "#B91C1C",
            }}
          >
            {this.state.error.message || String(this.state.error)}
          </p>
          <div className="flex items-center justify-center gap-2 mt-5">
            <button
              onClick={this.reset}
              className="h-9 px-4 rounded-xl flex items-center gap-2 text-white text-xs font-semibold transition-all"
              style={{
                background: "linear-gradient(135deg, #7C3AED, #3B82F6)",
                boxShadow: "0 4px 14px rgba(124,58,237,0.35)",
              }}
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Thử lại
            </button>
            <button
              onClick={() => window.location.reload()}
              className="h-9 px-4 rounded-xl flex items-center gap-2 text-xs font-medium transition-colors"
              style={{
                background: "rgba(124,58,237,0.08)",
                color: "#7C3AED",
                border: "1px solid rgba(124,58,237,0.15)",
              }}
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Tải lại trang
            </button>
          </div>
        </div>
      </div>
    );
  }
}

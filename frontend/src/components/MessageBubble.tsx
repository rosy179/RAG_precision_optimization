import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, ChevronDown, ChevronUp, FileText, Globe, Image } from "lucide-react";
import AiRobotIcon from "./AiRobotIcon";

export interface Source {
  rank: number;
  title: string;
  snippet: string;
  score: number;
  scope?: "session" | "global";
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  attachments?: string[];
  created_at?: string;
}

const typeIcon = (title: string) => {
  if (title.startsWith("http")) return <Globe className="w-3.5 h-3.5" />;
  if (/\.(png|jpg|jpeg|webp)/i.test(title)) return <Image className="w-3.5 h-3.5" />;
  return <FileText className="w-3.5 h-3.5" />;
};

export default function MessageBubble({ msg }: { msg: Message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[72%] flex flex-col items-end gap-1.5">
          {msg.attachments && msg.attachments.length > 0 && (
            <div className="flex flex-wrap justify-end gap-1.5">
              {msg.attachments.map((name, i) => (
                <span
                  key={i}
                  className="flex items-center gap-1.5 text-xs rounded-xl px-2.5 py-1.5 shadow-sm max-w-[240px]"
                  style={{
                    background: "rgba(255,255,255,0.7)",
                    border: "1px solid rgba(124,58,237,0.2)",
                    backdropFilter: "blur(8px)",
                  }}
                >
                  <span style={{ color: "#7C3AED" }} className="shrink-0">{typeIcon(name)}</span>
                  <span className="truncate text-[#1A1A2E]">{name}</span>
                </span>
              ))}
            </div>
          )}
          <div
            className="rounded-3xl rounded-tr-md px-4 py-3 text-sm leading-relaxed shadow-sm"
            style={{
              background: "linear-gradient(135deg, rgba(124,58,237,0.18) 0%, rgba(59,130,246,0.12) 100%)",
              border: "1px solid rgba(124,58,237,0.2)",
              backdropFilter: "blur(10px)",
              WebkitBackdropFilter: "blur(10px)",
              color: "#1A1A2E",
            }}
          >
            {msg.content}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 mb-5">
      {/* AI Avatar — mini robot */}
      <div className="shrink-0" style={{ marginTop: 2 }}>
        <AiRobotIcon mini />
      </div>

      <div className="flex-1 min-w-0">
        {/* Answer card */}
        <div
          className="rounded-3xl rounded-tl-md px-5 py-4 text-sm leading-relaxed"
          style={{
            background: "rgba(255,255,255,0.72)",
            backdropFilter: "blur(16px)",
            WebkitBackdropFilter: "blur(16px)",
            border: "1px solid rgba(255,255,255,0.85)",
            boxShadow: "0 4px 24px rgba(124,58,237,0.08)",
            color: "#1A1A2E",
          }}
        >
          <div className="prose prose-sm max-w-none prose-p:my-1.5 prose-headings:text-[#1A1A2E] prose-headings:mt-3 prose-headings:mb-1.5 prose-h3:text-sm prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5 prose-strong:text-[#1A1A2E] prose-table:text-xs prose-th:px-2 prose-th:py-1.5 prose-td:px-2 prose-td:py-1.5 prose-code:font-mono prose-code:text-xs prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded"
            style={{ ["--tw-prose-code-bg" as string]: "rgba(124,58,237,0.08)" }}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
          </div>
        </div>

        {/* Sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1.5 text-xs font-medium hover:underline transition-all"
              style={{ color: "#7C3AED" }}
            >
              {showSources ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              {showSources ? "Ẩn nguồn" : `${msg.sources.length} nguồn tham khảo`}
            </button>

            {showSources && (
              <div className="mt-2 space-y-1.5">
                {msg.sources.map((s) => (
                  <div
                    key={s.rank}
                    className="flex items-start gap-2 rounded-2xl px-3 py-2.5 shadow-sm"
                    style={{
                      background: "rgba(255,255,255,0.65)",
                      border: "1px solid rgba(124,58,237,0.15)",
                      backdropFilter: "blur(10px)",
                    }}
                  >
                    <div
                      className="w-5 h-5 rounded-lg flex items-center justify-center shrink-0 mt-0.5"
                      style={{ background: "rgba(124,58,237,0.1)", color: "#7C3AED" }}
                    >
                      {typeIcon(s.title)}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-semibold text-[#1A1A2E] truncate">{s.title}</p>
                      <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{s.snippet}</p>
                    </div>
                    {s.scope === "global" && (
                      <span
                        className="flex items-center gap-1 text-[10px] rounded-lg px-1.5 py-0.5 font-medium shrink-0"
                        style={{ background: "rgba(59,130,246,0.1)", color: "#3B82F6" }}
                        title="Từ kho kiến thức chung"
                      >
                        <BookOpen className="w-3 h-3" />
                        Kho chung
                      </span>
                    )}
                    <span
                      className="text-[10px] rounded-lg px-1.5 py-0.5 font-medium shrink-0"
                      style={{ background: "rgba(124,58,237,0.1)", color: "#7C3AED" }}
                    >
                      {(s.score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { ChevronDown, ChevronUp, FileText, Globe, Image, Bot } from "lucide-react";

export interface Source {
  rank: number;
  title: string;
  snippet: string;
  score: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
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
        <div className="max-w-[72%] bg-[#EDE9FE] rounded-3xl rounded-tr-md px-4 py-3 text-[#1A1A2E] text-sm leading-relaxed shadow-sm">
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 mb-5">
      {/* AI Avatar */}
      <div className="w-9 h-9 rounded-full bg-gradient-to-br from-[#7C3AED] to-[#3B82F6] flex items-center justify-center shrink-0 shadow">
        <Bot className="w-4 h-4 text-white" />
      </div>

      <div className="flex-1 min-w-0">
        {/* Answer card */}
        <div className="bg-white rounded-3xl rounded-tl-md px-5 py-4 shadow-card text-sm leading-relaxed text-[#1A1A2E]">
          <div className="prose prose-sm max-w-none prose-p:my-1 prose-headings:text-[#1A1A2E] prose-code:font-mono prose-code:text-xs prose-code:bg-[#F0EEFF] prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded">
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        </div>

        {/* Sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1.5 text-xs text-[#7C3AED] font-medium hover:underline"
            >
              {showSources ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              {showSources ? "Ẩn nguồn" : `${msg.sources.length} nguồn tham khảo`}
            </button>

            {showSources && (
              <div className="mt-2 space-y-1.5">
                {msg.sources.map((s) => (
                  <div key={s.rank} className="flex items-start gap-2 bg-white rounded-2xl px-3 py-2.5 shadow-sm border border-[#E0D9FF]">
                    <div className="w-5 h-5 rounded-lg bg-[#EDE9FE] flex items-center justify-center text-[#7C3AED] shrink-0 mt-0.5">
                      {typeIcon(s.title)}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-semibold text-[#1A1A2E] truncate">{s.title}</p>
                      <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{s.snippet}</p>
                    </div>
                    <span className="text-[10px] bg-[#F0EEFF] text-[#7C3AED] rounded-lg px-1.5 py-0.5 font-medium shrink-0">
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

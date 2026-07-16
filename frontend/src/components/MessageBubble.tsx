import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BookOpen, Check, ChevronDown, ChevronUp, Copy, FileText, GitBranch, Globe,
  Image, Loader2, RefreshCw, ThumbsDown, ThumbsUp,
} from "lucide-react";
import AiRobotIcon from "./AiRobotIcon";
import type { MultihopStep } from "../api/client";

export interface Source {
  rank: number;
  title: string;
  snippet: string;
  score: number;
  scope?: "session" | "global";
  doc_id?: string;
  chunk_id?: string;
  page?: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  attachments?: string[];
  feedback?: "up" | "down" | null;
  /** Multi-hop reasoning steps (chained retrieval) */
  steps?: MultihopStep[];
  /** true while SSE deltas are still arriving for this message */
  streaming?: boolean;
  created_at?: string;
}

interface Props {
  msg: Message;
  onFeedback?: (messageId: string, rating: "up" | "down" | null) => void;
  onRegenerate?: () => void;
  /** Open the document viewer at this source's location */
  onOpenSource?: (source: Source) => void;
}

const typeIcon = (title: string) => {
  if (title.startsWith("http")) return <Globe className="w-3.5 h-3.5" />;
  if (/\.(png|jpg|jpeg|webp)/i.test(title)) return <Image className="w-3.5 h-3.5" />;
  return <FileText className="w-3.5 h-3.5" />;
};

/** Turn bare [n] citation markers into markdown links (#cite-n) so the
 *  custom renderer below can style them as chips. Code spans/fences are
 *  left untouched ("arr[1]" in a snippet must not become a citation). */
const linkifyCitations = (md: string, maxRank: number): string => {
  if (maxRank < 1) return md;
  return md
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((seg, i) => {
      if (i % 2 === 1) return seg; // code segment — keep verbatim
      return seg.replace(/\[(\d+)\](?!\()/g, (m, n) => {
        const k = parseInt(n, 10);
        return k >= 1 && k <= maxRank ? `[${n}](#cite-${n})` : m;
      });
    })
    .join("");
};

function CitationChip({
  n, source, onClick,
}: { n: number; source?: Source; onClick: () => void }) {
  return (
    <span className="relative inline-block group" style={{ verticalAlign: "super", lineHeight: 1 }}>
      <button
        onClick={onClick}
        className="text-[10px] font-semibold rounded-md px-1 min-w-[16px] transition-colors"
        style={{
          background: "rgba(124,58,237,0.12)",
          color: "#7C3AED",
          border: "1px solid rgba(124,58,237,0.2)",
        }}
        title={source ? undefined : `Nguồn ${n}`}
      >
        {n}
      </button>
      {source && (
        <span
          className="hidden group-hover:block absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-72 rounded-xl px-3 py-2.5 text-left z-30 shadow-lg"
          style={{
            background: "rgba(255,255,255,0.97)",
            border: "1px solid rgba(124,58,237,0.2)",
            backdropFilter: "blur(12px)",
          }}
        >
          <span className="block text-[11px] font-semibold text-[#1A1A2E] truncate">
            {source.title}
          </span>
          <span className="block text-[11px] text-gray-500 mt-1 line-clamp-3 whitespace-normal">
            {source.snippet}
          </span>
        </span>
      )}
    </span>
  );
}

interface HopView {
  hop: number;
  total?: number;
  query?: string;
  answer?: string;
  titles?: string[];
}

/** Live view of the multi-hop reasoning chain: each hop shows the search
 *  query (often containing a fact discovered by the PREVIOUS hop) and the
 *  short bridging answer it produced. */
function MultihopSteps({ steps, streaming }: { steps: MultihopStep[]; streaming?: boolean }) {
  const [open, setOpen] = useState(false);
  const hops = useMemo<HopView[]>(() => {
    const map = new Map<number, HopView>();
    for (const s of steps) {
      const h = map.get(s.hop) ?? { hop: s.hop };
      if (s.stage === "hop_start") { h.query = s.query; h.total = s.total; }
      else { h.answer = s.answer; h.titles = s.titles; }
      map.set(s.hop, h);
    }
    return [...map.values()].sort((a, b) => a.hop - b.hop);
  }, [steps]);

  if (hops.length === 0) return null;
  const total = hops[0]?.total ?? hops.length;
  const expanded = streaming || open;

  return (
    <div
      className="mb-2 rounded-2xl overflow-hidden"
      style={{
        background: "rgba(124,58,237,0.05)",
        border: "1px solid rgba(124,58,237,0.18)",
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3.5 py-2 text-xs font-semibold transition-colors hover:bg-[rgba(124,58,237,0.06)]"
        style={{ color: "#7C3AED" }}
      >
        <GitBranch className="w-3.5 h-3.5 shrink-0" />
        {streaming
          ? `Đang suy luận nhiều bước (${hops.length}/${total})…`
          : `Suy luận qua ${hops.length} bước tìm kiếm`}
        {streaming
          ? <Loader2 className="w-3 h-3 animate-spin ml-auto" />
          : (expanded ? <ChevronUp className="w-3.5 h-3.5 ml-auto" /> : <ChevronDown className="w-3.5 h-3.5 ml-auto" />)}
      </button>

      {expanded && (
        <div className="px-3.5 pb-3 space-y-2">
          {hops.map((h) => (
            <div key={h.hop} className="flex items-start gap-2.5">
              <span
                className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5"
                style={{ background: "rgba(124,58,237,0.14)", color: "#7C3AED" }}
              >
                {h.hop}
              </span>
              <div className="min-w-0 flex-1 text-xs">
                <p className="text-[#1A1A2E] leading-snug">
                  <span className="font-medium text-gray-500">Tìm: </span>
                  {h.query}
                </p>
                {h.answer ? (
                  <p className="mt-1 leading-snug" style={{ color: "#047857" }}>
                    <span className="font-medium">↳ </span>
                    {h.answer}
                    {h.titles && h.titles.length > 0 && (
                      <span className="text-gray-400"> — {h.titles[0]}</span>
                    )}
                  </p>
                ) : (
                  <p className="mt-1 flex items-center gap-1.5 text-gray-400">
                    <Loader2 className="w-3 h-3 animate-spin" /> đang tìm…
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function MessageBubble({ msg, onFeedback, onRegenerate, onOpenSource }: Props) {
  const [showSources, setShowSources] = useState(false);
  const [highlightRank, setHighlightRank] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const highlightTimer = useRef<number>();
  const isUser = msg.role === "user";

  useEffect(() => () => window.clearTimeout(highlightTimer.current), []);

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

  const jumpToSource = (rank: number) => {
    setShowSources(true);
    setHighlightRank(rank);
    window.clearTimeout(highlightTimer.current);
    highlightTimer.current = window.setTimeout(() => setHighlightRank(null), 1800);
  };

  const copyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(msg.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard unavailable */ }
  };

  // Feedback needs the server-side message id (uuid). Locally-created ids
  // (Date.now()) mean the message never persisted — hide the buttons.
  const canRate = !!onFeedback && msg.id.includes("-") && !msg.streaming;
  const rendered = linkifyCitations(msg.content, msg.sources?.length ?? 0);

  return (
    <div className="flex items-start gap-3 mb-5">
      {/* AI Avatar — mini robot */}
      <div className="shrink-0" style={{ marginTop: 2 }}>
        <AiRobotIcon mini />
      </div>

      <div className="flex-1 min-w-0">
        {/* Multi-hop reasoning chain (if the router chose chained retrieval) */}
        {msg.steps && msg.steps.length > 0 && (
          <MultihopSteps steps={msg.steps} streaming={msg.streaming} />
        )}

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
          {msg.streaming && !msg.content ? (
            <div className="flex items-center gap-1.5 py-1">
              {[0, 150, 300].map((delay) => (
                <div
                  key={delay}
                  className="w-2 h-2 rounded-full"
                  style={{
                    background: "linear-gradient(135deg, #7C3AED, #3B82F6)",
                    animation: "dot-bounce 1.2s ease-in-out infinite",
                    animationDelay: `${delay}ms`,
                  }}
                />
              ))}
            </div>
          ) : (
            <div className="prose prose-sm max-w-none prose-p:my-1.5 prose-headings:text-[#1A1A2E] prose-headings:mt-3 prose-headings:mb-1.5 prose-h3:text-sm prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5 prose-strong:text-[#1A1A2E] prose-table:text-xs prose-th:px-2 prose-th:py-1.5 prose-td:px-2 prose-td:py-1.5 prose-code:font-mono prose-code:text-xs prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded"
              style={{ ["--tw-prose-code-bg" as string]: "rgba(124,58,237,0.08)" }}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: ({ href, children }) => {
                    if (href?.startsWith("#cite-")) {
                      const n = parseInt(href.slice(6), 10);
                      const src = msg.sources?.find((s) => s.rank === n) ?? msg.sources?.[n - 1];
                      return (
                        <CitationChip
                          n={n}
                          source={src}
                          onClick={() => {
                            // Old messages have no doc_id — fall back to the card list
                            if (src?.doc_id && onOpenSource) onOpenSource(src);
                            else jumpToSource(n);
                          }}
                        />
                      );
                    }
                    return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
                  },
                }}
              >
                {rendered}
              </ReactMarkdown>
              {msg.streaming && (
                <span
                  className="inline-block w-2 h-4 ml-0.5 rounded-sm align-text-bottom animate-pulse"
                  style={{ background: "linear-gradient(135deg, #7C3AED, #3B82F6)" }}
                />
              )}
            </div>
          )}
        </div>

        {/* Action row: copy / rate / regenerate */}
        {!msg.streaming && msg.content && (
          <div className="flex items-center gap-1 mt-1.5 ml-1">
            <button
              onClick={copyAnswer}
              title="Sao chép câu trả lời"
              className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-400 hover:text-[#7C3AED] hover:bg-[rgba(124,58,237,0.08)] transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5" style={{ color: "#10B981" }} /> : <Copy className="w-3.5 h-3.5" />}
            </button>
            {canRate && (
              <>
                <button
                  onClick={() => onFeedback!(msg.id, msg.feedback === "up" ? null : "up")}
                  title="Câu trả lời hữu ích"
                  className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors hover:bg-[rgba(16,185,129,0.1)]"
                  style={{ color: msg.feedback === "up" ? "#10B981" : "#9CA3AF" }}
                >
                  <ThumbsUp className="w-3.5 h-3.5" fill={msg.feedback === "up" ? "currentColor" : "none"} />
                </button>
                <button
                  onClick={() => onFeedback!(msg.id, msg.feedback === "down" ? null : "down")}
                  title="Câu trả lời chưa tốt"
                  className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors hover:bg-[rgba(239,68,68,0.1)]"
                  style={{ color: msg.feedback === "down" ? "#EF4444" : "#9CA3AF" }}
                >
                  <ThumbsDown className="w-3.5 h-3.5" fill={msg.feedback === "down" ? "currentColor" : "none"} />
                </button>
              </>
            )}
            {onRegenerate && (
              <button
                onClick={onRegenerate}
                title="Tạo lại câu trả lời"
                className="h-7 px-2 rounded-lg flex items-center gap-1.5 text-[11px] font-medium text-gray-400 hover:text-[#7C3AED] hover:bg-[rgba(124,58,237,0.08)] transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Tạo lại
              </button>
            )}
          </div>
        )}

        {/* Sources */}
        {msg.sources && msg.sources.length > 0 && !msg.streaming && (
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
                    onClick={() => { if (s.doc_id && onOpenSource) onOpenSource(s); }}
                    role={s.doc_id && onOpenSource ? "button" : undefined}
                    title={s.doc_id && onOpenSource ? "Mở tài liệu tại đoạn này" : undefined}
                    className={`flex items-start gap-2 rounded-2xl px-3 py-2.5 shadow-sm transition-all ${
                      s.doc_id && onOpenSource ? "cursor-pointer hover:shadow-md" : ""
                    }`}
                    style={{
                      background: highlightRank === s.rank
                        ? "rgba(124,58,237,0.1)"
                        : "rgba(255,255,255,0.65)",
                      border: highlightRank === s.rank
                        ? "1px solid rgba(124,58,237,0.45)"
                        : "1px solid rgba(124,58,237,0.15)",
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
                      <p className="text-xs font-semibold text-[#1A1A2E] truncate">
                        <span style={{ color: "#7C3AED" }}>[{s.rank}]</span> {s.title}
                        {s.page && (
                          <span className="ml-1.5 font-normal text-gray-400">· tr.{s.page}</span>
                        )}
                      </p>
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

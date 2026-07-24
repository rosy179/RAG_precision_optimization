import { useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import {
  AlignLeft, ChevronLeft, ChevronRight, FileText, Loader2, X,
} from "lucide-react";
import { documentsAPI, knowledgeAPI, authHeaders } from "../api/client";
import type { DocContent } from "../api/client";
import { useI18n } from "../i18n/LanguageProvider";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

export interface ViewerTarget {
  docId: string;
  scope: "session" | "global";
  title?: string;
  page?: number;
  chunkId?: string;
  snippet?: string;
}

const escapeHtml = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Progressively shorter prefixes of the snippet used to find the cited
 *  passage inside pdf.js text items (items are line fragments, so long
 *  phrases often span two items and never match). */
const highlightPhrases = (snippet?: string): string[] => {
  if (!snippet) return [];
  const words = snippet.trim().split(/\s+/);
  return [8, 5, 3]
    .filter((n) => words.length >= n)
    .map((n) => words.slice(0, n).join(" ").toLowerCase());
};

export default function DocViewerPanel({
  target, onClose,
}: { target: ViewerTarget; onClose: () => void }) {
  const { t } = useI18n();
  const [content, setContent] = useState<DocContent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"pdf" | "text">("text");
  const [numPages, setNumPages] = useState(0);
  const [pageNumber, setPageNumber] = useState(1);
  const [pageWidth, setPageWidth] = useState(480);
  const bodyRef = useRef<HTMLDivElement>(null);
  const markRef = useRef<HTMLElement | null>(null);

  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const api = target.scope === "global" ? knowledgeAPI : documentsAPI;

  // react-pdf reloads the document whenever the `file` prop identity
  // changes — keep it stable per docId.
  const pdfFile = useMemo(
    () => ({ url: api.fileUrl(target.docId), httpHeaders: authHeaders() }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [target.docId, target.scope],
  );

  useEffect(() => {
    let cancelled = false;
    setContent(null);
    setError(null);
    api.content(target.docId)
      .then((data) => {
        if (cancelled) return;
        setContent(data);
        const chunk = data.chunks.find((c) => c.id === target.chunkId);
        const page = target.page ?? chunk?.page ?? 1;
        setPageNumber(page);
        setMode(data.has_file ? "pdf" : "text");
      })
      .catch((e) => {
        if (!cancelled) setError(e?.response?.data?.detail || t("viewer.loadError"));
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target.docId, target.scope, target.chunkId, target.page]);

  // Fit PDF pages to the panel width
  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const update = () => setPageWidth(Math.max(280, el.clientWidth - 32));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Scroll the highlighted passage into view in text mode.
  // Only scroll the panel body itself: scrollIntoView() also adjusts every
  // scrollable ancestor — including the overflow-hidden chat/app containers,
  // which have no scrollbars, so the whole layout ends up shifted with no
  // way for the user to scroll it back.
  useEffect(() => {
    if (mode === "text" && content) {
      const t = window.setTimeout(() => {
        const body = bodyRef.current;
        const mark = markRef.current;
        if (!body) return;
        if (!mark) { body.scrollTop = 0; return; }
        // Align the START of the passage near the top — cited chunks are
        // often taller than the panel, so centering lands mid-passage.
        const offset = mark.getBoundingClientRect().top - body.getBoundingClientRect().top;
        body.scrollTo({
          top: Math.max(0, body.scrollTop + offset - 64),
          behavior: "smooth",
        });
      }, 60);
      return () => window.clearTimeout(t);
    }
  }, [mode, content, target.chunkId]);

  const phrases = useMemo(() => highlightPhrases(target.snippet), [target.snippet]);

  const textRenderer = useMemo(() => {
    if (!phrases.length) return undefined;
    return ({ str }: { str: string }) => {
      const lower = str.toLowerCase();
      for (const p of phrases) {
        const idx = lower.indexOf(p);
        if (idx !== -1) {
          return (
            escapeHtml(str.slice(0, idx)) +
            `<mark style="background:rgba(250,204,21,0.55);border-radius:2px;">` +
            escapeHtml(str.slice(idx, idx + p.length)) +
            "</mark>" +
            escapeHtml(str.slice(idx + p.length))
          );
        }
      }
      return escapeHtml(str);
    };
  }, [phrases]);

  const renderText = () => {
    if (!content) return null;
    const chunk = content.chunks.find((c) => c.id === target.chunkId);
    let start = chunk?.start ?? -1;
    let end = chunk?.end ?? -1;
    if (start < 0 && target.snippet) {
      // No chunk offsets (e.g. old message) — locate the snippet directly
      const probe = target.snippet.slice(0, 120).trim();
      const idx = probe ? content.text.indexOf(probe) : -1;
      if (idx !== -1) { start = idx; end = idx + probe.length; }
    }
    if (start < 0 || end <= start) {
      return <p className="whitespace-pre-wrap">{content.text}</p>;
    }
    return (
      <p className="whitespace-pre-wrap">
        {content.text.slice(0, start)}
        <mark
          ref={(el) => { markRef.current = el; }}
          style={{
            background: "rgba(250,204,21,0.45)",
            borderRadius: 4,
            padding: "1px 2px",
          }}
        >
          {content.text.slice(start, end)}
        </mark>
        {content.text.slice(end)}
      </p>
    );
  };

  return (
    <div
      className="flex flex-col h-full"
      style={{
        background: "rgba(255,255,255,0.92)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        borderLeft: "1px solid rgba(124,58,237,0.15)",
        boxShadow: "-8px 0 32px rgba(124,58,237,0.12)",
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 px-4 py-3 shrink-0"
        style={{ borderBottom: "1px solid rgba(124,58,237,0.12)" }}
      >
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
          style={{ background: "rgba(124,58,237,0.1)", color: "#7C3AED" }}
        >
          <FileText className="w-4 h-4" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-[#1A1A2E] truncate">
            {content?.name || target.title || t("common.document")}
          </p>
          {target.scope === "global" && (
            <p className="text-[10px]" style={{ color: "#3B82F6" }}>{t("viewer.globalKb")}</p>
          )}
        </div>
        {content?.has_file && (
          <button
            onClick={() => setMode(mode === "pdf" ? "text" : "pdf")}
            title={mode === "pdf" ? t("viewer.viewText") : t("viewer.viewPdf")}
            className="h-7 px-2 rounded-lg flex items-center gap-1 text-[11px] font-medium transition-colors"
            style={{
              background: "rgba(124,58,237,0.08)",
              color: "#7C3AED",
              border: "1px solid rgba(124,58,237,0.15)",
            }}
          >
            {mode === "pdf" ? <AlignLeft className="w-3.5 h-3.5" /> : <FileText className="w-3.5 h-3.5" />}
            {mode === "pdf" ? t("viewer.text") : t("viewer.pdf")}
          </button>
        )}
        <button
          onClick={onClose}
          className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-black/5 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Cited snippet */}
      {target.snippet && (
        <div
          className="mx-4 mt-3 px-3 py-2 rounded-xl text-[11px] shrink-0"
          style={{
            background: "rgba(250,204,21,0.12)",
            border: "1px solid rgba(250,204,21,0.4)",
            color: "#713F12",
          }}
        >
          <span className="font-semibold">{t("viewer.citedSnippet")}</span>
          <span className="line-clamp-2">{target.snippet}</span>
        </div>
      )}

      {/* Body */}
      <div ref={bodyRef} className="flex-1 overflow-y-auto px-4 py-3 text-[13px] leading-relaxed text-[#1A1A2E]">
        {error && (
          <p className="text-sm text-red-500 mt-4 text-center">{error}</p>
        )}
        {!error && !content && (
          <div className="flex items-center justify-center gap-2 mt-8 text-gray-400 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" /> {t("viewer.loading")}
          </div>
        )}
        {content && mode === "text" && renderText()}
        {content && mode === "pdf" && (
          <Document
            file={pdfFile}
            onLoadSuccess={({ numPages: n }) => {
              setNumPages(n);
              setPageNumber((p) => Math.min(Math.max(1, p), n));
            }}
            onLoadError={() => setMode("text")}
            loading={
              <div className="flex items-center justify-center gap-2 mt-8 text-gray-400 text-sm">
                <Loader2 className="w-4 h-4 animate-spin" /> {t("viewer.loadingPdf")}
              </div>
            }
          >
            <Page
              pageNumber={pageNumber}
              width={pageWidth}
              customTextRenderer={textRenderer}
              renderAnnotationLayer={false}
            />
          </Document>
        )}
      </div>

      {/* PDF page nav */}
      {content && mode === "pdf" && numPages > 0 && (
        <div
          className="flex items-center justify-center gap-3 px-4 py-2 shrink-0"
          style={{ borderTop: "1px solid rgba(124,58,237,0.12)" }}
        >
          <button
            onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
            disabled={pageNumber <= 1}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-500 hover:bg-black/5 disabled:opacity-30 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-xs text-gray-600 font-medium">
            {t("viewer.pageOf", { page: pageNumber, total: numPages })}
          </span>
          <button
            onClick={() => setPageNumber((p) => Math.min(numPages, p + 1))}
            disabled={pageNumber >= numPages}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-500 hover:bg-black/5 disabled:opacity-30 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}

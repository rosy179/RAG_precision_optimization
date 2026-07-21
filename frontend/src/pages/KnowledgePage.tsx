import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle, BookOpen, CheckCircle2, Eye, FileSpreadsheet, FileText, Globe,
  Image, Layers, Link2, Loader2, Mic, Presentation, RefreshCw, Search, Trash2, Upload, X,
} from "lucide-react";
import { knowledgeAPI } from "../api/client";
import type { KbDoc } from "../api/client";
import { useI18n } from "../i18n/LanguageProvider";
import DocViewerPanel from "../components/DocViewerPanel";
import ChunkManagerPanel from "../components/ChunkManagerPanel";
import type { ViewerTarget } from "../components/DocViewerPanel";

const FILE_ACCEPT = ".pdf,.txt,.md,.markdown,.docx,.pptx,.xlsx,.png,.jpg,.jpeg,.webp,.mp3,.wav,.m4a,.ogg,.webm";

interface UploadStatus {
  id: string;
  name: string;
  status: "uploading" | "done" | "error";
  detail?: string;
  /** Upload progress 0–100 (bytes sent); 100 = server chunking/indexing. */
  progress?: number;
}

const typeIcon = (type: string) => {
  if (type === "url") return <Globe className="w-4 h-4" />;
  if (type === "image") return <Image className="w-4 h-4" />;
  if (type === "audio") return <Mic className="w-4 h-4" />;
  if (type === "xlsx") return <FileSpreadsheet className="w-4 h-4" />;
  if (type === "pptx") return <Presentation className="w-4 h-4" />;
  return <FileText className="w-4 h-4" />;
};

// Format names stay verbatim across languages; only the descriptive kinds
// (text/web/image/audio) are translated via the component's `typeLabel`.
const TYPE_STATIC: Record<string, string> = {
  pdf: "PDF", markdown: "Markdown", docx: "Word", pptx: "PowerPoint",
  xlsx: "Excel", corpus: "Corpus", json: "JSON",
};

export default function KnowledgePage() {
  const { t } = useI18n();
  const typeLabel = (type: string): string =>
    TYPE_STATIC[type]
    ?? ({ txt: t("kb.type.text"), url: t("kb.type.web"), image: t("kb.type.image"), audio: t("kb.type.audio") }[type])
    ?? type;
  const [docs, setDocs] = useState<KbDoc[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [canManage, setCanManage] = useState(false);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [uploads, setUploads] = useState<UploadStatus[]>([]);
  const [urlInput, setUrlInput] = useState("");
  const [reloading, setReloading] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [viewerTarget, setViewerTarget] = useState<ViewerTarget | null>(null);
  const [chunkTarget, setChunkTarget] = useState<{ id: string; name: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    knowledgeAPI.list()
      .then((data) => {
        setDocs(data.documents || []);
        setTotalChunks(data.total_chunks || 0);
        setCanManage(!!data.can_manage);
      })
      .catch(() => { setDocs([]); setTotalChunks(0); })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const uploadFiles = useCallback(async (files: File[]) => {
    for (const file of files) {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      setUploads((p) => [...p, { id, name: file.name, status: "uploading", progress: 0 }]);
      try {
        const res = await knowledgeAPI.uploadFile(file, (pct) =>
          setUploads((p) => p.map((u) => (u.id === id ? { ...u, progress: pct } : u)))
        );
        setUploads((p) => p.map((u) =>
          u.id === id ? { ...u, status: "done" as const, progress: 100, detail: `${res.chunk_count} chunks` } : u
        ));
        load();
      } catch (e: any) {
        setUploads((p) => p.map((u) =>
          u.id === id ? { ...u, status: "error" as const, detail: e?.response?.data?.detail || t("kb.uploadError") } : u
        ));
      }
    }
  }, [load, t]);

  const uploadUrl = useCallback(async () => {
    const url = urlInput.trim();
    if (!/^https?:\/\/\S+$/i.test(url)) return;
    setUrlInput("");
    const id = `${Date.now()}-url`;
    setUploads((p) => [...p, { id, name: url, status: "uploading" }]);
    try {
      const res = await knowledgeAPI.uploadUrl(url);
      setUploads((p) => p.map((u) =>
        u.id === id ? { ...u, name: res.name || url, status: "done" as const, detail: `${res.chunk_count} chunks` } : u
      ));
      load();
    } catch (e: any) {
      setUploads((p) => p.map((u) =>
        u.id === id ? { ...u, status: "error" as const, detail: e?.response?.data?.detail || t("kb.urlError") } : u
      ));
    }
  }, [urlInput, load, t]);

  const doDelete = useCallback(async (docId: string) => {
    setConfirmDelete(null);
    setDeleting(docId);
    try {
      await knowledgeAPI.delete(docId);
      if (viewerTarget?.docId === docId) setViewerTarget(null);
      load();
    } catch { /* keep the row; next load() reflects reality */ }
    setDeleting(null);
  }, [load, viewerTarget]);

  const doReload = useCallback(async () => {
    setReloading(true);
    try {
      await knowledgeAPI.reload();
      load();
    } catch { /* ignore */ }
    setReloading(false);
  }, [load]);

  const shown = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return docs;
    return docs.filter((d) => d.name.toLowerCase().includes(q));
  }, [docs, filter]);

  return (
    <div
      className="flex flex-1 h-full overflow-hidden"
      style={{
        background: "rgba(255,255,255,0.3)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={FILE_ACCEPT}
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files || []);
          if (files.length) uploadFiles(files);
          e.target.value = "";
        }}
      />

      <div className="flex flex-col flex-1 min-w-0 px-4 py-5 sm:px-6 md:px-10 md:py-8 pt-16 md:pt-8 overflow-y-auto">
        {/* Header */}
        <div className="flex flex-wrap items-start justify-between gap-3 sm:gap-4 mb-5 md:mb-6">
          <div className="flex items-center gap-3">
            <div
              className="w-11 h-11 rounded-2xl flex items-center justify-center"
              style={{ background: "linear-gradient(135deg, rgba(59,130,246,0.15), rgba(124,58,237,0.15))", color: "#3B82F6" }}
            >
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-[#1A1A2E]">{t("kb.title")}</h1>
              <p className="text-xs text-gray-500 mt-0.5">
                {t("kb.subtitle", { docs: docs.length, chunks: totalChunks.toLocaleString() })}
              </p>
            </div>
          </div>
          {canManage && (
            <button
              onClick={doReload}
              disabled={reloading}
              title={t("kb.syncTitle")}
              className="h-9 px-3 rounded-xl flex items-center gap-2 text-xs font-medium transition-colors disabled:opacity-50"
              style={{
                background: "rgba(124,58,237,0.08)",
                color: "#7C3AED",
                border: "1px solid rgba(124,58,237,0.15)",
              }}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${reloading ? "animate-spin" : ""}`} />
              {t("kb.sync")}
            </button>
          )}
        </div>

        {/* Upload zone (admins only) */}
        {canManage && (
          <div
            className="rounded-2xl px-5 py-4 mb-5"
            style={{
              background: "rgba(255,255,255,0.6)",
              border: "1px dashed rgba(124,58,237,0.35)",
            }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const files = Array.from(e.dataTransfer.files);
              if (files.length) uploadFiles(files);
            }}
          >
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="h-9 px-4 rounded-xl flex items-center gap-2 text-white text-xs font-semibold transition-all"
                style={{
                  background: "linear-gradient(135deg, #7C3AED, #3B82F6)",
                  boxShadow: "0 4px 14px rgba(124,58,237,0.35)",
                }}
              >
                <Upload className="w-3.5 h-3.5" />
                {t("kb.upload")}
              </button>
              <div className="flex items-center flex-1 min-w-[240px] gap-2">
                <div
                  className="flex items-center gap-2 flex-1 h-9 px-3 rounded-xl"
                  style={{ background: "rgba(255,255,255,0.8)", border: "1px solid rgba(124,58,237,0.15)" }}
                >
                  <Link2 className="w-3.5 h-3.5 shrink-0" style={{ color: "#3B82F6" }} />
                  <input
                    className="flex-1 bg-transparent outline-none text-xs text-[#1A1A2E] placeholder-gray-400"
                    placeholder={t("kb.urlPlaceholder")}
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") uploadUrl(); }}
                  />
                </div>
                <button
                  onClick={uploadUrl}
                  disabled={!/^https?:\/\/\S+$/i.test(urlInput.trim())}
                  className="h-9 px-3 rounded-xl text-xs font-medium transition-colors disabled:opacity-40"
                  style={{
                    background: "rgba(59,130,246,0.1)",
                    color: "#3B82F6",
                    border: "1px solid rgba(59,130,246,0.2)",
                  }}
                >
                  {t("kb.addUrl")}
                </button>
              </div>
              <p className="w-full text-[11px] text-gray-400">
                {t("kb.uploadHint")}
              </p>
            </div>

            {uploads.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {uploads.map((u) => (
                  <span
                    key={u.id}
                    className="relative overflow-hidden flex items-center gap-1.5 text-xs rounded-xl px-3 py-1.5 max-w-[300px]"
                    style={
                      u.status === "error"
                        ? { background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#EF4444" }
                        : { background: "rgba(124,58,237,0.07)", border: "1px solid rgba(124,58,237,0.15)", color: "#1A1A2E" }
                    }
                  >
                    {u.status === "uploading" && <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" style={{ color: "#7C3AED" }} />}
                    {u.status === "done" && <CheckCircle2 className="w-3.5 h-3.5 shrink-0" style={{ color: "#10B981" }} />}
                    {u.status === "error" && <AlertCircle className="w-3.5 h-3.5 shrink-0" />}
                    <span className="truncate">{u.name}</span>
                    {u.status === "uploading"
                      ? <span className="text-gray-400 shrink-0">· {(u.progress ?? 0) < 100 ? `${u.progress ?? 0}%` : t("common.processing")}</span>
                      : u.detail && <span className="text-gray-400 shrink-0">· {u.detail}</span>}
                    {u.status !== "uploading" && (
                      <button
                        onClick={() => setUploads((p) => p.filter((x) => x.id !== u.id))}
                        className="text-gray-400 hover:text-gray-600 shrink-0"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    )}
                    {u.status === "uploading" && (
                      <span
                        className="absolute bottom-0 left-0 h-0.5 transition-all duration-200"
                        style={{ width: `${u.progress ?? 0}%`, background: "#7C3AED" }}
                      />
                    )}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Search */}
        <div
          className="flex items-center gap-2 h-10 px-3.5 rounded-2xl mb-4"
          style={{ background: "rgba(255,255,255,0.7)", border: "1px solid rgba(124,58,237,0.12)" }}
        >
          <Search className="w-4 h-4 text-gray-400 shrink-0" />
          <input
            className="flex-1 bg-transparent outline-none text-sm text-[#1A1A2E] placeholder-gray-400"
            placeholder={t("kb.searchPlaceholder")}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          {filter && (
            <button onClick={() => setFilter("")} className="text-gray-400 hover:text-gray-600">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Docs list */}
        {loading ? (
          <div className="flex items-center justify-center gap-2 mt-10 text-gray-400 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" /> {t("common.loading")}
          </div>
        ) : shown.length === 0 ? (
          <p className="text-sm text-gray-400 text-center mt-10">
            {filter ? t("kb.noMatch") : t("kb.empty")}
          </p>
        ) : (
          <div className="space-y-1.5 pb-8">
            {shown.map((d) => (
              <div
                key={d.id}
                className="group flex items-center gap-3 rounded-2xl px-4 py-3 transition-all hover:shadow-md cursor-pointer"
                style={{
                  background: "rgba(255,255,255,0.65)",
                  border: "1px solid rgba(124,58,237,0.12)",
                }}
                onClick={() => setViewerTarget({ docId: d.id, scope: "global", title: d.name })}
                title={t("kb.viewContent")}
              >
                <div
                  className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
                  style={{ background: "rgba(124,58,237,0.08)", color: "#7C3AED" }}
                >
                  {typeIcon(d.type)}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-[#1A1A2E] truncate">{d.name}</p>
                  <p className="text-[11px] text-gray-400 mt-0.5">
                    {typeLabel(d.type)} · {d.chunk_count} chunks
                    {d.created_at && ` · ${new Date(d.created_at).toLocaleDateString()}`}
                  </p>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); setViewerTarget({ docId: d.id, scope: "global", title: d.name }); }}
                  title={t("kb.viewContentShort")}
                  className="hidden md:flex opacity-0 group-hover:opacity-100 w-8 h-8 rounded-lg items-center justify-center text-gray-400 hover:text-[#7C3AED] hover:bg-[rgba(124,58,237,0.08)] transition-all shrink-0"
                >
                  <Eye className="w-4 h-4" />
                </button>
                {canManage && (
                  <button
                    onClick={(e) => { e.stopPropagation(); setChunkTarget({ id: d.id, name: d.name }); }}
                    title={t("kb.editChunk")}
                    className="hidden md:flex opacity-0 group-hover:opacity-100 w-8 h-8 rounded-lg items-center justify-center text-gray-400 hover:text-[#7C3AED] hover:bg-[rgba(124,58,237,0.08)] transition-all shrink-0"
                  >
                    <Layers className="w-4 h-4" />
                  </button>
                )}
                {canManage && (confirmDelete === d.id ? (
                  <div className="flex items-center gap-1.5 shrink-0" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => doDelete(d.id)}
                      className="h-8 px-2.5 rounded-lg text-[11px] font-semibold text-white"
                      style={{ background: "#EF4444" }}
                    >
                      {t("common.deleteHard")}
                    </button>
                    <button
                      onClick={() => setConfirmDelete(null)}
                      className="h-8 px-2 rounded-lg text-[11px] text-gray-500 hover:bg-black/5"
                    >
                      {t("common.cancel")}
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={(e) => { e.stopPropagation(); setConfirmDelete(d.id); }}
                    title={t("kb.delete")}
                    disabled={deleting === d.id}
                    className="opacity-100 md:opacity-0 md:group-hover:opacity-100 w-8 h-8 rounded-lg flex items-center justify-center text-gray-400 hover:text-red-500 hover:bg-red-50 transition-all shrink-0 disabled:opacity-100"
                  >
                    {deleting === d.id
                      ? <Loader2 className="w-4 h-4 animate-spin" />
                      : <Trash2 className="w-4 h-4" />}
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Content viewer — same fixed side drawer as the chat page */}
      {viewerTarget && (
        <>
          <div
            aria-label={t("viewer.close")}
            onClick={() => setViewerTarget(null)}
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(15,10,30,0.22)",
              backdropFilter: "blur(2px)",
              WebkitBackdropFilter: "blur(2px)",
              zIndex: 40,
              animation: "fadeIn 0.2s ease",
            }}
          />
          <div
            className="fixed inset-y-0 right-0 z-[41] flex flex-col overflow-hidden w-full sm:w-[min(70vw,520px)] lg:w-[min(44vw,560px)]"
            style={{ animation: "slideInRight 0.28s cubic-bezier(0.4,0,0.2,1)" }}
          >
            <DocViewerPanel target={viewerTarget} onClose={() => setViewerTarget(null)} />
          </div>
        </>
      )}

      {/* Chunk editor — same drawer pattern (admin) */}
      {chunkTarget && (
        <>
          <div
            aria-label={t("chunk.closeTitle")}
            onClick={() => setChunkTarget(null)}
            style={{
              position: "fixed", inset: 0, background: "rgba(15,10,30,0.22)",
              backdropFilter: "blur(2px)", WebkitBackdropFilter: "blur(2px)",
              zIndex: 40, animation: "fadeIn 0.2s ease",
            }}
          />
          <div
            className="fixed inset-y-0 right-0 z-[41] flex flex-col overflow-hidden w-full sm:w-[min(70vw,520px)] lg:w-[min(44vw,560px)]"
            style={{ animation: "slideInRight 0.28s cubic-bezier(0.4,0,0.2,1)" }}
          >
            <ChunkManagerPanel
              docId={chunkTarget.id}
              docName={chunkTarget.name}
              onClose={() => setChunkTarget(null)}
              onChanged={load}
            />
          </div>
        </>
      )}
    </div>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle, BookOpen, CheckCircle2, Eye, FileText, Globe, Image, Link2,
  Loader2, Mic, RefreshCw, Search, Trash2, Upload, X,
} from "lucide-react";
import { knowledgeAPI } from "../api/client";
import type { KbDoc } from "../api/client";
import DocViewerPanel from "../components/DocViewerPanel";
import type { ViewerTarget } from "../components/DocViewerPanel";

const FILE_ACCEPT = ".pdf,.txt,.png,.jpg,.jpeg,.webp,.mp3,.wav,.m4a,.ogg,.webm";

interface UploadStatus {
  id: string;
  name: string;
  status: "uploading" | "done" | "error";
  detail?: string;
}

const typeIcon = (type: string) => {
  if (type === "url") return <Globe className="w-4 h-4" />;
  if (type === "image") return <Image className="w-4 h-4" />;
  if (type === "audio") return <Mic className="w-4 h-4" />;
  return <FileText className="w-4 h-4" />;
};

const typeLabel: Record<string, string> = {
  pdf: "PDF", txt: "Văn bản", url: "Web", image: "Ảnh", audio: "Âm thanh",
  corpus: "Corpus", json: "JSON",
};

export default function KnowledgePage() {
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
      setUploads((p) => [...p, { id, name: file.name, status: "uploading" }]);
      try {
        const res = await knowledgeAPI.uploadFile(file);
        setUploads((p) => p.map((u) =>
          u.id === id ? { ...u, status: "done" as const, detail: `${res.chunk_count} chunks` } : u
        ));
        load();
      } catch (e: any) {
        setUploads((p) => p.map((u) =>
          u.id === id ? { ...u, status: "error" as const, detail: e?.response?.data?.detail || "Lỗi upload" } : u
        ));
      }
    }
  }, [load]);

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
        u.id === id ? { ...u, status: "error" as const, detail: e?.response?.data?.detail || "Không thể fetch URL" } : u
      ));
    }
  }, [urlInput, load]);

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

      <div className="flex flex-col flex-1 min-w-0 px-10 py-8 overflow-y-auto">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <div
              className="w-11 h-11 rounded-2xl flex items-center justify-center"
              style={{ background: "linear-gradient(135deg, rgba(59,130,246,0.15), rgba(124,58,237,0.15))", color: "#3B82F6" }}
            >
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-[#1A1A2E]">Kho kiến thức chung</h1>
              <p className="text-xs text-gray-500 mt-0.5">
                {docs.length} tài liệu · {totalChunks.toLocaleString("vi-VN")} chunks · dùng chung cho mọi cuộc trò chuyện
              </p>
            </div>
          </div>
          {canManage && (
            <button
              onClick={doReload}
              disabled={reloading}
              title="Đọc lại kho từ đĩa (sau khi chạy scripts/ingest_knowledge.py)"
              className="h-9 px-3 rounded-xl flex items-center gap-2 text-xs font-medium transition-colors disabled:opacity-50"
              style={{
                background: "rgba(124,58,237,0.08)",
                color: "#7C3AED",
                border: "1px solid rgba(124,58,237,0.15)",
              }}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${reloading ? "animate-spin" : ""}`} />
              Đồng bộ từ đĩa
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
                Tải tệp lên
              </button>
              <div className="flex items-center flex-1 min-w-[240px] gap-2">
                <div
                  className="flex items-center gap-2 flex-1 h-9 px-3 rounded-xl"
                  style={{ background: "rgba(255,255,255,0.8)", border: "1px solid rgba(124,58,237,0.15)" }}
                >
                  <Link2 className="w-3.5 h-3.5 shrink-0" style={{ color: "#3B82F6" }} />
                  <input
                    className="flex-1 bg-transparent outline-none text-xs text-[#1A1A2E] placeholder-gray-400"
                    placeholder="Dán URL bài viết rồi Enter…"
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
                  Thêm URL
                </button>
              </div>
              <p className="w-full text-[11px] text-gray-400">
                PDF, TXT, ảnh, ghi âm hoặc kéo-thả tệp vào đây. Tài liệu sẽ được chia chunk, tóm tắt và đánh index ngay — không cần khởi động lại server.
              </p>
            </div>

            {uploads.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {uploads.map((u) => (
                  <span
                    key={u.id}
                    className="flex items-center gap-1.5 text-xs rounded-xl px-3 py-1.5 max-w-[300px]"
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
                    {u.detail && <span className="text-gray-400 shrink-0">· {u.detail}</span>}
                    {u.status !== "uploading" && (
                      <button
                        onClick={() => setUploads((p) => p.filter((x) => x.id !== u.id))}
                        className="text-gray-400 hover:text-gray-600 shrink-0"
                      >
                        <X className="w-3 h-3" />
                      </button>
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
            placeholder="Tìm tài liệu theo tên…"
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
            <Loader2 className="w-4 h-4 animate-spin" /> Đang tải…
          </div>
        ) : shown.length === 0 ? (
          <p className="text-sm text-gray-400 text-center mt-10">
            {filter ? "Không có tài liệu nào khớp." : "Kho kiến thức đang trống."}
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
                title="Xem nội dung tài liệu"
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
                    {typeLabel[d.type] || d.type} · {d.chunk_count} chunks
                    {d.created_at && ` · ${new Date(d.created_at).toLocaleDateString("vi-VN")}`}
                  </p>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); setViewerTarget({ docId: d.id, scope: "global", title: d.name }); }}
                  title="Xem nội dung"
                  className="opacity-0 group-hover:opacity-100 w-8 h-8 rounded-lg flex items-center justify-center text-gray-400 hover:text-[#7C3AED] hover:bg-[rgba(124,58,237,0.08)] transition-all shrink-0"
                >
                  <Eye className="w-4 h-4" />
                </button>
                {canManage && (confirmDelete === d.id ? (
                  <div className="flex items-center gap-1.5 shrink-0" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => doDelete(d.id)}
                      className="h-8 px-2.5 rounded-lg text-[11px] font-semibold text-white"
                      style={{ background: "#EF4444" }}
                    >
                      Xóa hẳn
                    </button>
                    <button
                      onClick={() => setConfirmDelete(null)}
                      className="h-8 px-2 rounded-lg text-[11px] text-gray-500 hover:bg-black/5"
                    >
                      Hủy
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={(e) => { e.stopPropagation(); setConfirmDelete(d.id); }}
                    title="Xóa khỏi kho chung"
                    disabled={deleting === d.id}
                    className="opacity-0 group-hover:opacity-100 w-8 h-8 rounded-lg flex items-center justify-center text-gray-400 hover:text-red-500 hover:bg-red-50 transition-all shrink-0 disabled:opacity-100"
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
            aria-label="Đóng tài liệu"
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
            style={{
              position: "fixed",
              top: 0,
              right: 0,
              bottom: 0,
              width: "min(44%, 560px)",
              zIndex: 41,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
              animation: "slideInRight 0.28s cubic-bezier(0.4,0,0.2,1)",
            }}
          >
            <DocViewerPanel target={viewerTarget} onClose={() => setViewerTarget(null)} />
          </div>
        </>
      )}
    </div>
  );
}

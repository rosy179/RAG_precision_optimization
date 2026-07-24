import { useEffect, useState } from "react";
import { Check, Loader2, Pencil, Save, Trash2, X } from "lucide-react";
import { knowledgeAPI } from "../api/client";
import type { KbChunk } from "../api/client";
import { useI18n } from "../i18n/LanguageProvider";

/** Admin chunk editor for a KB document (RAGFlow-style): list every parsed
 *  chunk, edit its text (re-embeds), or delete a junk chunk. Fits the
 *  project's "precision" positioning — the retrieved context is only as good
 *  as the chunks in the store. */
export default function ChunkManagerPanel({
  docId, docName, onClose, onChanged,
}: {
  docId: string;
  docName: string;
  onClose: () => void;
  /** Called after any edit/delete so the doc list can refresh chunk_count. */
  onChanged?: () => void;
}) {
  const { t } = useI18n();
  const [chunks, setChunks] = useState<KbChunk[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmDel, setConfirmDel] = useState<string | null>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape" && !editing) onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose, editing]);

  const load = () => {
    setChunks(null);
    setError(null);
    knowledgeAPI.chunks(docId)
      .then((d) => setChunks(d.chunks))
      .catch((e) => setError(e?.response?.data?.detail || t("chunk.loadError")));
  };
  useEffect(load, [docId]);

  const startEdit = (c: KbChunk) => { setEditing(c.id); setDraft(c.text); };

  const saveEdit = async (c: KbChunk) => {
    const text = draft.trim();
    if (!text || text === c.text) { setEditing(null); return; }
    setBusy(c.id);
    try {
      await knowledgeAPI.updateChunk(docId, c.id, text);
      setChunks((prev) => prev?.map((x) => (x.id === c.id ? { ...x, text } : x)) ?? null);
      setEditing(null);
      onChanged?.();
    } catch (e: any) {
      setError(e?.response?.data?.detail || t("chunk.saveError"));
    } finally {
      setBusy(null);
    }
  };

  const doDelete = async (c: KbChunk) => {
    setBusy(c.id);
    try {
      await knowledgeAPI.deleteChunk(docId, c.id);
      setChunks((prev) => prev?.filter((x) => x.id !== c.id) ?? null);
      setConfirmDel(null);
      onChanged?.();
    } catch (e: any) {
      setError(e?.response?.data?.detail || t("chunk.deleteError"));
    } finally {
      setBusy(null);
    }
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
      <div className="flex items-center gap-2 px-4 py-3 shrink-0"
        style={{ borderBottom: "1px solid rgba(124,58,237,0.12)" }}>
        <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
          style={{ background: "rgba(124,58,237,0.1)", color: "#7C3AED" }}>
          <Pencil className="w-4 h-4" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-[#1A1A2E] truncate">{t("chunk.title", { name: docName })}</p>
          <p className="text-[10px]" style={{ color: "#7C3AED" }}>
            {chunks ? t("chunk.count", { n: chunks.length }) : t("common.loading")} · {t("chunk.subtitle")}
          </p>
        </div>
        <button
          onClick={onClose}
          className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-black/5 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2.5">
        {error && (
          <div className="rounded-xl px-3 py-2 text-xs"
            style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", color: "#B91C1C" }}>
            {error}
          </div>
        )}
        {!error && !chunks && (
          <div className="flex items-center justify-center gap-2 mt-8 text-gray-400 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" /> {t("chunk.loading")}
          </div>
        )}
        {chunks && chunks.length === 0 && (
          <p className="text-sm text-gray-400 text-center mt-8">{t("chunk.empty")}</p>
        )}
        {chunks?.map((c) => (
          <div key={c.id} className="rounded-2xl overflow-hidden"
            style={{ background: "rgba(255,255,255,0.7)", border: "1px solid rgba(124,58,237,0.14)" }}>
            <div className="flex items-center gap-2 px-3 py-1.5"
              style={{ background: "rgba(124,58,237,0.05)" }}>
              <span className="text-[10px] font-bold rounded px-1.5 py-0.5"
                style={{ background: "rgba(124,58,237,0.12)", color: "#7C3AED" }}>
                #{c.chunk_index}
              </span>
              {c.page && <span className="text-[10px] text-gray-400">{t("msg.page", { page: c.page })}</span>}
              <div className="ml-auto flex items-center gap-1">
                {editing === c.id ? (
                  <>
                    <button
                      onClick={() => saveEdit(c)}
                      disabled={busy === c.id}
                      title={t("common.save")}
                      className="h-6 px-2 rounded-lg flex items-center gap-1 text-[11px] font-semibold text-white transition-colors disabled:opacity-50"
                      style={{ background: "#7C3AED" }}
                    >
                      {busy === c.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                      {t("common.save")}
                    </button>
                    <button onClick={() => setEditing(null)}
                      className="h-6 px-2 rounded-lg text-[11px] text-gray-500 hover:bg-black/5">
                      {t("common.cancel")}
                    </button>
                  </>
                ) : confirmDel === c.id ? (
                  <>
                    <button
                      onClick={() => doDelete(c)}
                      disabled={busy === c.id}
                      className="h-6 px-2 rounded-lg flex items-center gap-1 text-[11px] font-semibold text-white disabled:opacity-50"
                      style={{ background: "#EF4444" }}
                    >
                      {busy === c.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                      {t("common.deleteHard")}
                    </button>
                    <button onClick={() => setConfirmDel(null)}
                      className="h-6 px-2 rounded-lg text-[11px] text-gray-500 hover:bg-black/5">
                      {t("common.cancel")}
                    </button>
                  </>
                ) : (
                  <>
                    <button onClick={() => startEdit(c)} title={t("chunk.edit")}
                      className="w-6 h-6 rounded-lg flex items-center justify-center text-gray-400 hover:text-[#7C3AED] hover:bg-[rgba(124,58,237,0.08)] transition-colors">
                      <Pencil className="w-3.5 h-3.5" />
                    </button>
                    <button onClick={() => setConfirmDel(c.id)} title={t("chunk.delete")}
                      className="w-6 h-6 rounded-lg flex items-center justify-center text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}
              </div>
            </div>
            {editing === c.id ? (
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                autoFocus
                rows={Math.min(12, Math.max(4, Math.ceil(draft.length / 60)))}
                className="w-full px-3 py-2 text-[12px] leading-relaxed text-[#1A1A2E] resize-y outline-none bg-transparent"
                style={{ fontFamily: "inherit" }}
              />
            ) : (
              <p className="px-3 py-2 text-[12px] leading-relaxed text-[#1A1A2E] whitespace-pre-wrap">
                {c.text}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

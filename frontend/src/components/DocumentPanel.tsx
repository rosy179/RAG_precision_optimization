import { useEffect, useState, useCallback } from "react";
import { Globe, FileText, Image, Trash2, AudioLines, Paperclip } from "lucide-react";
import { documentsAPI } from "../api/client";

interface Doc { id: string; name: string; type: string; chunk_count: number; created_at: string; }

const typeIcon = (type: string) => {
  if (type === "url") return <Globe className="w-4 h-4 text-[#3B82F6]" />;
  if (type === "image") return <Image className="w-4 h-4 text-purple-500" />;
  if (type === "audio") return <AudioLines className="w-4 h-4 text-indigo-400" />;
  return <FileText className="w-4 h-4 text-[#7C3AED]" />;
};

export default function DocumentPanel({ onDocsChange }: { onDocsChange?: () => void }) {
  const [docs, setDocs] = useState<Doc[]>([]);

  const refreshDocs = useCallback(() => {
    documentsAPI.list().then((d) => setDocs(d.documents || [])).catch(() => {});
  }, []);

  useEffect(() => { refreshDocs(); }, []);

  const handleDelete = async (id: string) => {
    await documentsAPI.delete(id).catch(() => {});
    refreshDocs();
    onDocsChange?.();
  };

  return (
    <div className="flex flex-col gap-4">
      {docs.length === 0 ? (
        <div className="text-center py-8 px-2">
          <Paperclip className="w-6 h-6 text-[#C4B5FD] mx-auto mb-3" />
          <p className="text-xs text-gray-400 leading-relaxed">
            Chưa có tài liệu nào.<br />
            Đính kèm tệp, dán ảnh / URL hoặc ghi âm
            ngay tại ô chat để upload.
          </p>
        </div>
      ) : (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
            Tài liệu ({docs.length})
          </p>
          <div className="space-y-1.5">
            {docs.map((d) => (
              <div key={d.id} className="flex items-center gap-2 group bg-white rounded-xl px-3 py-2 shadow-sm border border-[#E0D9FF]">
                <div className="shrink-0">{typeIcon(d.type)}</div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-[#1A1A2E] truncate">{d.name}</p>
                  <p className="text-[10px] text-gray-400">{d.chunk_count} chunks</p>
                </div>
                <button
                  onClick={() => handleDelete(d.id)}
                  className="opacity-0 group-hover:opacity-100 text-gray-300 hover:text-red-400 transition-all"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

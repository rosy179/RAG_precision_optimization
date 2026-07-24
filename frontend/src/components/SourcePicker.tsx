import type { Dispatch, SetStateAction } from "react";
import { BookOpen, FileText, X } from "lucide-react";
import { useI18n } from "../i18n/LanguageProvider";

interface PickerDoc {
  id: string;
  name: string;
  chunk_count: number;
}

/**
 * Source-picker popover extracted from ChatPage (E3): lets the user choose
 * which session documents + the shared KB participate in the next question.
 * Presentational — state lives in the parent and drives `sourceParams()`.
 */
export default function SourcePicker({
  sessionDocs, kbDocCount, excludedDocs, setExcludedDocs,
  useGlobalKb, setUseGlobalKb, onClose,
}: {
  sessionDocs: PickerDoc[];
  kbDocCount: number;
  excludedDocs: Set<string>;
  setExcludedDocs: Dispatch<SetStateAction<Set<string>>>;
  useGlobalKb: boolean;
  setUseGlobalKb: (v: boolean) => void;
  onClose: () => void;
}) {
  const { t } = useI18n();
  return (
    <div
      className="mb-3 rounded-2xl px-3 py-3"
      style={{
        background: "rgba(255,255,255,0.9)",
        border: "1px solid rgba(124,58,237,0.18)",
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold text-[#1A1A2E]">{t("picker.title")}</p>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      <label className="flex items-center gap-2 text-xs text-[#1A1A2E] py-1.5 cursor-pointer">
        <input
          type="checkbox"
          checked={useGlobalKb}
          onChange={(e) => setUseGlobalKb(e.target.checked)}
          className="accent-[#7C3AED]"
        />
        <BookOpen className="w-3.5 h-3.5" style={{ color: "#3B82F6" }} />
        <span>{t("picker.globalKb")} <span className="text-gray-400">{t("picker.docCount", { n: kbDocCount })}</span></span>
      </label>
      {sessionDocs.length > 0 ? (
        <div className="max-h-40 overflow-y-auto mt-1 space-y-0.5">
          {sessionDocs.map((d) => (
            <label key={d.id} className="flex items-center gap-2 text-xs text-[#1A1A2E] py-1 cursor-pointer">
              <input
                type="checkbox"
                checked={!excludedDocs.has(d.id)}
                onChange={(e) => {
                  setExcludedDocs((prev) => {
                    const next = new Set(prev);
                    if (e.target.checked) next.delete(d.id);
                    else next.add(d.id);
                    return next;
                  });
                }}
                className="accent-[#7C3AED]"
              />
              <FileText className="w-3.5 h-3.5 shrink-0" style={{ color: "#7C3AED" }} />
              <span className="truncate flex-1">{d.name}</span>
              <span className="text-gray-400 shrink-0">{d.chunk_count} chunks</span>
            </label>
          ))}
        </div>
      ) : (
        <p className="text-[11px] text-gray-400 mt-1">
          {t("picker.empty")}
        </p>
      )}
    </div>
  );
}

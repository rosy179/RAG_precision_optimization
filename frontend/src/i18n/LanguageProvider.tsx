import {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
  type ReactNode,
} from "react";
import { dictionaries, type Lang, type TKey } from "./translations";

const STORAGE_KEY = "lang";

function detectInitial(): Lang {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === "vi" || saved === "en" || saved === "ja") return saved;
  const nav = navigator.language.toLowerCase();
  if (nav.startsWith("ja")) return "ja";
  if (nav.startsWith("en")) return "en";
  return "vi"; // app's home language
}

/** Translate a key, substituting {name} placeholders from `params`. */
export type TFunc = (key: TKey, params?: Record<string, string | number>) => string;

interface Ctx {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: TFunc;
}

const LanguageContext = createContext<Ctx | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(detectInitial);

  useEffect(() => { document.documentElement.lang = lang; }, [lang]);

  const setLang = useCallback((l: Lang) => {
    localStorage.setItem(STORAGE_KEY, l);
    setLangState(l);
  }, []);

  const t = useCallback<TFunc>((key, params) => {
    let s: string = dictionaries[lang][key] ?? dictionaries.vi[key] ?? key;
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        s = s.split(`{${k}}`).join(String(v));
      }
    }
    return s;
  }, [lang]);

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useI18n(): Ctx {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error("useI18n must be used within a LanguageProvider");
  return ctx;
}

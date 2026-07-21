import { useState, useCallback, useEffect } from "react";
import { authAPI } from "../api/client";

export interface AuthUser { user_id: string; email: string; name: string; is_admin?: boolean; }

const persist = (user: AuthUser) => localStorage.setItem("user", JSON.stringify(user));

export function useAuth() {
  const [user, setUser] = useState<AuthUser | null>(() => {
    const stored = localStorage.getItem("user");
    return stored ? JSON.parse(stored) : null;
  });

  // Refresh server-side attributes (is_admin) on load so sessions predating
  // the flag — or a changed ADMIN_EMAILS — reflect the current permissions.
  useEffect(() => {
    if (!localStorage.getItem("token")) return;
    authAPI.me()
      .then(({ is_admin }) => setUser((u) => {
        if (!u || u.is_admin === is_admin) return u;
        const next = { ...u, is_admin };
        persist(next);
        return next;
      }))
      .catch(() => { /* interceptor handles 401; keep cached user otherwise */ });
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const data = await authAPI.login(email, password);
    localStorage.setItem("token", data.access_token);
    const next: AuthUser = { user_id: data.user_id, email: data.email, name: data.name || "", is_admin: data.is_admin };
    persist(next);
    setUser(next);
  }, []);

  const register = useCallback(async (email: string, password: string, name: string) => {
    const data = await authAPI.register(email, password, name);
    localStorage.setItem("token", data.access_token);
    const next: AuthUser = { user_id: data.user_id, email: data.email, name, is_admin: data.is_admin };
    persist(next);
    setUser(next);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setUser(null);
  }, []);

  return { user, login, register, logout };
}

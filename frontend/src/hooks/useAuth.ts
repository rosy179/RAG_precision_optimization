import { useState, useCallback } from "react";
import { authAPI } from "../api/client";

export interface AuthUser { user_id: string; email: string; name: string; }

export function useAuth() {
  const [user, setUser] = useState<AuthUser | null>(() => {
    const stored = localStorage.getItem("user");
    return stored ? JSON.parse(stored) : null;
  });

  const login = useCallback(async (email: string, password: string) => {
    const data = await authAPI.login(email, password);
    localStorage.setItem("token", data.access_token);
    localStorage.setItem("user", JSON.stringify({ user_id: data.user_id, email: data.email, name: data.name || "" }));
    setUser({ user_id: data.user_id, email: data.email, name: data.name || "" });
  }, []);

  const register = useCallback(async (email: string, password: string, name: string) => {
    const data = await authAPI.register(email, password, name);
    localStorage.setItem("token", data.access_token);
    localStorage.setItem("user", JSON.stringify({ user_id: data.user_id, email: data.email, name }));
    setUser({ user_id: data.user_id, email: data.email, name });
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setUser(null);
  }, []);

  return { user, login, register, logout };
}

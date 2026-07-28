import { useCallback, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  AuthContext,
  ACCESS_KEY_STORAGE_KEY,
  REVIEWER_ID_STORAGE_KEY,
  readSession,
} from "./AuthContext";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKeyState] = useState<string>(() => readSession(ACCESS_KEY_STORAGE_KEY));
  const [reviewerId, setReviewerIdState] = useState<string>(() => readSession(REVIEWER_ID_STORAGE_KEY));

  const setApiKey = useCallback((key: string) => {
    const trimmed = key.trim();
    if (typeof window !== "undefined") {
      window.sessionStorage.setItem(ACCESS_KEY_STORAGE_KEY, trimmed);
    }
    setApiKeyState(trimmed);
  }, []);

  const setReviewerId = useCallback((id: string) => {
    const trimmed = id.trim();
    if (typeof window !== "undefined") {
      window.sessionStorage.setItem(REVIEWER_ID_STORAGE_KEY, trimmed);
    }
    setReviewerIdState(trimmed);
  }, []);

  const clearApiKey = useCallback(() => {
    if (typeof window !== "undefined") {
      window.sessionStorage.removeItem(ACCESS_KEY_STORAGE_KEY);
    }
    setApiKeyState("");
  }, []);

  const value = useMemo(
    () => ({ apiKey, reviewerId, setApiKey, setReviewerId, clearApiKey }),
    [apiKey, reviewerId, setApiKey, setReviewerId, clearApiKey]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

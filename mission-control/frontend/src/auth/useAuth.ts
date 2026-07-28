import { useContext } from "react";
import { AuthContext } from "./AuthContext";
import type { AuthValue } from "./AuthContext";

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext);
  if (ctx === null) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return ctx;
}

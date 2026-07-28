/** Shared auth context type and object. Split from the provider component so
 * the provider file exports only a component (keeps React fast-refresh working)
 * and this file holds no JSX. */

import { createContext } from "react";

export const ACCESS_KEY_STORAGE_KEY = "mission_control_api_key";
export const REVIEWER_ID_STORAGE_KEY = "mission_control_reviewer_id";

export function readSession(key: string): string {
  if (typeof window === "undefined") return "";
  return window.sessionStorage.getItem(key)?.trim() ?? "";
}

export interface AuthValue {
  apiKey: string;
  reviewerId: string;
  /** Store a validated access key. */
  setApiKey: (key: string) => void;
  /** Store the reviewer identity used for decision provenance. */
  setReviewerId: (id: string) => void;
  /** Drop the access key (e.g. after a 401/1008) and re-lock the console. */
  clearApiKey: () => void;
}

export const AuthContext = createContext<AuthValue | null>(null);

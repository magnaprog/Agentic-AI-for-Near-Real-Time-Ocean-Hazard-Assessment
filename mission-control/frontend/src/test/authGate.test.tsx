/**
 * The unlock screen is the only place a reviewer ID can be supplied, and
 * useApi rejects every review decision without one. These tests pin that
 * contract from both directions: the gate must refuse a blank reviewer ID,
 * and a session that somehow holds a blank one must be sent back to the gate
 * rather than into a console where every decision fails on submit.
 */
import { render, screen, fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import AuthGate from "../components/AuthGate";
import { AuthProvider } from "../auth/AuthProvider";
import { ACCESS_KEY_STORAGE_KEY, REVIEWER_ID_STORAGE_KEY } from "../auth/AuthContext";
import { useAuth } from "../auth/useAuth";

function GateHarness() {
  const { apiKey, reviewerId } = useAuth();
  const locked = apiKey === "" || reviewerId === "";
  return locked ? <AuthGate /> : <div>console unlocked</div>;
}

function renderGate() {
  return render(
    <AuthProvider>
      <GateHarness />
    </AuthProvider>
  );
}

describe("AuthGate", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("refuses to unlock without an access key", () => {
    renderGate();
    fireEvent.click(screen.getByRole("button", { name: /unlock console/i }));
    expect(screen.getByRole("alert")).toHaveTextContent(/enter the access key/i);
    expect(screen.queryByText("console unlocked")).toBeNull();
  });

  it("refuses to unlock without a reviewer ID", () => {
    renderGate();
    fireEvent.change(screen.getByLabelText(/access key/i), { target: { value: "k" } });
    fireEvent.click(screen.getByRole("button", { name: /unlock console/i }));
    expect(screen.getByRole("alert")).toHaveTextContent(/reviewer id/i);
    expect(window.sessionStorage.getItem(ACCESS_KEY_STORAGE_KEY)).toBeNull();
    expect(screen.queryByText("console unlocked")).toBeNull();
  });

  it("stores both values and unlocks when each is supplied", () => {
    renderGate();
    fireEvent.change(screen.getByLabelText(/access key/i), { target: { value: " k " } });
    fireEvent.change(screen.getByLabelText(/reviewer id/i), { target: { value: " ada " } });
    fireEvent.click(screen.getByRole("button", { name: /unlock console/i }));
    expect(screen.getByText("console unlocked")).toBeTruthy();
    expect(window.sessionStorage.getItem(ACCESS_KEY_STORAGE_KEY)).toBe("k");
    expect(window.sessionStorage.getItem(REVIEWER_ID_STORAGE_KEY)).toBe("ada");
  });

  it("re-locks a stored session whose reviewer ID is blank", () => {
    // The state a previous build could leave behind: a usable access key with
    // no reviewer ID. Keying the gate on the access key alone would open the
    // console here and strand the operator.
    window.sessionStorage.setItem(ACCESS_KEY_STORAGE_KEY, "k");
    window.sessionStorage.setItem(REVIEWER_ID_STORAGE_KEY, "");
    renderGate();
    expect(screen.queryByText("console unlocked")).toBeNull();
    expect(screen.getByRole("dialog")).toBeTruthy();
  });

  it("stops announcing a field as invalid once the operator types", () => {
    // aria-invalid was derived from the error message and nothing cleared it,
    // so a screen reader kept calling the field invalid while it was fixed.
    renderGate();
    const accessKey = screen.getByLabelText(/access key/i);
    fireEvent.click(screen.getByRole("button", { name: /unlock console/i }));
    expect(accessKey.getAttribute("aria-invalid")).toBe("true");

    fireEvent.change(accessKey, { target: { value: "k" } });
    expect(accessKey.getAttribute("aria-invalid")).toBe("false");
    expect(screen.getByRole("alert")).toHaveTextContent("");
  });

  it("wraps Tab from the last control back to the first", () => {
    renderGate();
    const submit = screen.getByRole("button", { name: /unlock console/i });
    const accessKey = screen.getByLabelText(/access key/i);
    submit.focus();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Tab" });
    expect(document.activeElement).toBe(accessKey);
  });

  it("wraps Shift+Tab from the first control back to the last", () => {
    renderGate();
    const submit = screen.getByRole("button", { name: /unlock console/i });
    const accessKey = screen.getByLabelText(/access key/i);
    accessKey.focus();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(submit);
  });

  it("does not claim modality it cannot enforce", () => {
    // No focus trap is implemented, so aria-modal would hide the console from
    // assistive technology while Tab still reached it.
    renderGate();
    expect(screen.getByRole("dialog").getAttribute("aria-modal")).toBeNull();
  });
});

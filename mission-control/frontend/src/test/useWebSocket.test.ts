/**
 * Behavioral spec for the WebSocket re-lock policy.
 *
 * The safety property under test: a rejected key always re-locks the console
 * (fail-closed), while a BFF that is merely down or restarting never discards
 * a valid key behind a false "session expired". A close before onopen is
 * ambiguous (the BFF refuses a bad key at the HTTP handshake, which the
 * browser surfaces as an abnormal close), so the hook disambiguates with one
 * authenticated probe: 401 re-locks, anything else keeps the key and retries.
 */
import { renderHook, act, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useWebSocket } from "../hooks/useWebSocket";
import { AUTH_REQUIRED_EVENT, probeAuthRejected } from "../hooks/useApi";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  close() {
    this.closed = true;
  }
}

function lastSocket(): FakeWebSocket {
  return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
}

describe("useWebSocket re-lock policy", () => {
  let authEvents: number;
  const onAuthRequired = () => {
    authEvents += 1;
  };

  beforeEach(() => {
    authEvents = 0;
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
  });

  afterEach(() => {
    window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("re-locks immediately on an application-level 1008 close", async () => {
    vi.stubGlobal("fetch", vi.fn());
    const { unmount } = renderHook(() => useWebSocket("/ws", "some-key"));
    act(() => {
      lastSocket().onopen?.();
      lastSocket().onclose?.({ code: 1008 });
    });
    expect(authEvents).toBe(1);
    // No probe for an explicit rejection.
    expect(vi.mocked(fetch)).not.toHaveBeenCalled();
    unmount();
  });

  it("re-locks after a close-before-open only when the probe confirms 401", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ status: 401 } as Response),
    );
    const { unmount } = renderHook(() => useWebSocket("/ws", "bad-key"));
    act(() => {
      // Close without ever opening: the ambiguous case.
      lastSocket().onclose?.({ code: 1006 });
    });
    await waitFor(() => expect(authEvents).toBe(1));
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1);
    unmount();
  });

  it("keeps the key and reconnects when the probe cannot reach the BFF", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("network down")),
    );
    const { unmount } = renderHook(() => useWebSocket("/ws", "good-key"));
    const before = FakeWebSocket.instances.length;
    await act(async () => {
      lastSocket().onclose?.({ code: 1006 });
      // Let the probe promise settle before switching to fake timers.
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(authEvents).toBe(0);
    // A reconnect is scheduled rather than a re-lock: after the 1s base
    // backoff elapses, a second socket is constructed.
    await waitFor(
      () => expect(FakeWebSocket.instances.length).toBeGreaterThan(before),
      { timeout: 3000 },
    );
    expect(authEvents).toBe(0);
    unmount();
  });

  it("keeps the key when the probe reaches the BFF and the key is accepted", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ status: 200 } as Response),
    );
    const { unmount } = renderHook(() => useWebSocket("/ws", "good-key"));
    await act(async () => {
      lastSocket().onclose?.({ code: 1006 });
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(authEvents).toBe(0);
    unmount();
  });
});

describe("probeAuthRejected classification", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns true only for 401", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ status: 401 } as Response));
    expect(await probeAuthRejected("k")).toBe(true);
  });

  it("returns false for any other server answer", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ status: 503 } as Response));
    expect(await probeAuthRejected("k")).toBe(false);
  });

  it("returns null when the server is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("refused")));
    expect(await probeAuthRejected("k")).toBe(null);
  });
});

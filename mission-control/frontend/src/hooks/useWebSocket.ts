import { useEffect, useRef, useState } from "react";
import type { WSMessage, SystemSnapshot } from "../types";
import { probeAuthRejected, signalAuthRequired } from "./useApi";

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

export function useWebSocket(baseUrl: string, apiKey: string) {
  const [snapshot, setSnapshot] = useState<SystemSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  // True once a socket has opened for the current key. `connected` starts
  // false, so on its own it cannot tell "the first handshake is still in
  // flight" from "an established link dropped". Callers were painting a red
  // CONNECTION LOST alert on every page load and every unlock because of it.
  const [everConnected, setEverConnected] = useState(false);
  const [upstreamError, setUpstreamError] = useState(false);
  // Local receipt time of the last message proving the BFF reached the core
  // API. Measured on this clock, not the BFF's, so browser/server skew cannot
  // make a healthy link read as stale.
  const [lastContactMs, setLastContactMs] = useState<number | null>(null);
  const backoffRef = useRef(RECONNECT_BASE_MS);

  useEffect(() => {
    if (apiKey === "") {
      // Locked: no socket. Reset so a stale view is not shown after re-lock.
      setSnapshot(null);
      setConnected(false);
      setEverConnected(false);
      setUpstreamError(false);
      setLastContactMs(null);
      return;
    }

    // A new key or endpoint is a fresh link: nothing has been established yet.
    setEverConnected(false);

    let cancelled = false;
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    backoffRef.current = RECONNECT_BASE_MS;

    const connect = () => {
      if (cancelled) return;
      // The key rides in a subprotocol, not the query string. A browser
      // cannot set request headers on a WebSocket, and a query parameter is
      // written verbatim into the uvicorn access log on every handshake and
      // every reconnect. Sec-WebSocket-Protocol is not part of that log line.
      // base64url because a subprotocol value must be an HTTP token.
      const encodedKey = btoa(
        String.fromCharCode(...new TextEncoder().encode(apiKey))
      )
        .replace(/\+/g, "-")
        .replace(/\//g, "_")
        .replace(/=+$/, "");
      ws = new WebSocket(baseUrl, [`mc-key.${encodedKey}`]);
      let opened = false;

      ws.onopen = () => {
        if (cancelled) return;
        opened = true;
        setConnected(true);
        setEverConnected(true);
        backoffRef.current = RECONNECT_BASE_MS;
      };

      ws.onmessage = (event) => {
        if (cancelled) return;
        try {
          const msg: WSMessage = JSON.parse(event.data);
          if (msg.type === "snapshot") {
            setSnapshot(msg.data);
            setUpstreamError(false);
            // Deliberately not a contact signal: connect() replays the retained
            // snapshot even during an outage, which would read as a fresh poll.
            // Only the heartbeat, which the BFF sends after a poll succeeds,
            // moves the staleness clock.
          } else if (msg.type === "heartbeat") {
            setLastContactMs(Date.now());
            setUpstreamError(false);
          } else if (msg.type === "upstream_error") {
            setUpstreamError(true);
          } else if (msg.type === "upstream_recovered") {
            setUpstreamError(false);
          }
        } catch {
          /* ignore malformed messages */
        }
      };

      const scheduleReconnect = () => {
        timer = setTimeout(() => {
          if (!cancelled) connect();
        }, backoffRef.current);
        backoffRef.current = Math.min(backoffRef.current * 2, RECONNECT_MAX_MS);
      };

      ws.onclose = (event) => {
        if (cancelled) return;
        setConnected(false);
        // An application-level 1008 is the BFF explicitly rejecting the key
        // after accepting the socket: re-lock immediately.
        if (event.code === 1008) {
          setSnapshot(null);
          signalAuthRequired();
          return;
        }
        // A close before onopen is ambiguous. The BFF refuses a bad key at
        // the HTTP handshake (403, surfaced as an abnormal 1006), but a BFF
        // that is down or restarting closes the same way. Probe once with an
        // authenticated GET: a confirmed 401 re-locks; anything else keeps
        // the key and retries with backoff, so a transient outage does not
        // discard a valid key behind a false "session expired" note. A wrong
        // key still can never unlock the console: once the BFF is reachable
        // the probe returns 401 and the gate comes back.
        if (!opened) {
          void probeAuthRejected(apiKey).then((rejected) => {
            if (cancelled) return;
            if (rejected === true) {
              setSnapshot(null);
              signalAuthRequired();
            } else {
              scheduleReconnect();
            }
          });
          return;
        }
        // Drop of an already-open socket: plain reconnect.
        scheduleReconnect();
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      clearTimeout(timer);
      if (ws) {
        // Detach handlers so this closing socket cannot schedule a reconnect
        // after the effect is gone (e.g. during a key change or re-lock).
        ws.onopen = null;
        ws.onmessage = null;
        ws.onclose = null;
        ws.onerror = null;
        ws.close();
      }
    };
  }, [baseUrl, apiKey]);

  return { snapshot, connected, everConnected, upstreamError, lastContactMs };
}

import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent as ReactKeyboardEvent } from "react";
import { useAuth } from "../auth/useAuth";

interface Props {
  /** Shown when the previous key was rejected, to explain the re-lock. */
  expired?: boolean;
}

/** Unlock screen for the console. Replaces a browser prompt with a stable form
 * so the operator can paste an access key (and reviewer ID) without the page
 * blocking behind window.prompt. Submits on Enter. */
export default function AuthGate({ expired = false }: Props) {
  const { setApiKey, setReviewerId } = useAuth();
  const [key, setKey] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [error, setError] = useState("");
  const keyRef = useRef<HTMLInputElement>(null);
  const gateRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    keyRef.current?.focus();
  }, []);

  // Keep Tab inside the gate. The console is still rendered and focusable
  // behind the overlay, so without this a keyboard user tabs off the submit
  // button into controls they cannot see and cannot use while locked.
  const handleKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "Tab") {
      return;
    }
    const focusable = gateRef.current?.querySelectorAll<HTMLElement>(
      "input:not([disabled]), button:not([disabled])"
    );
    if (focusable === undefined || focusable.length === 0) {
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && active === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmedKey = key.trim();
    if (trimmedKey === "") {
      setError("Enter the access key to continue.");
      return;
    }
    // Required, not optional: the console never asks again, and useApi rejects
    // every review decision when the reviewer ID is empty. Accepting a blank
    // one here would leave the operator unable to record a decision all session.
    const trimmedReviewer = reviewer.trim();
    if (trimmedReviewer === "") {
      setError("Enter a reviewer ID; it attributes every review decision.");
      return;
    }
    setReviewerId(trimmedReviewer);
    setApiKey(trimmedKey);
  };

  // Tab is trapped, but aria-modal stays off: the console behind the overlay is
  // not marked inert, so a screen reader's virtual cursor can still reach it.
  // Claiming modality would promise more isolation than this provides.
  return (
    <div
      className="gate"
      ref={gateRef}
      role="dialog"
      aria-labelledby="gate-title"
      onKeyDown={handleKeyDown}
    >
      {/* noValidate: the inputs are marked required so assistive tech
          announces them as such, but validation is handled here so the
          message lands in the one alert region both fields point at
          through aria-describedby, rather than in a browser tooltip. */}
      <form className="gate__card" noValidate onSubmit={handleSubmit}>
        <div className="gate__title" id="gate-title">
          Mission Control access
        </div>
        <p className="gate__sub">
          {expired
            ? "The stored access key was rejected. Enter a current key to reconnect."
            : "Enter the access key provided by operations to open the console."}
        </p>

        <div className="gate__field">
          <label className="gate__label" htmlFor="gate-key">
            Access key
          </label>
          <input
            id="gate-key"
            ref={keyRef}
            className="gate__input"
            type="password"
            autoComplete="off"
            required
            aria-invalid={error !== "" && key.trim() === ""}
            aria-describedby="gate-error"
            value={key}
            onChange={(e) => setKey(e.target.value)}
          />
        </div>

        <div className="gate__field">
          <label className="gate__label" htmlFor="gate-reviewer">
            Reviewer ID <span style={{ textTransform: "none", letterSpacing: 0 }}>(required; attributes your review decisions)</span>
          </label>
          <input
            id="gate-reviewer"
            className="gate__input"
            type="text"
            autoComplete="off"
            required
            aria-invalid={error !== "" && key.trim() !== "" && reviewer.trim() === ""}
            aria-describedby="gate-error"
            placeholder="name or badge"
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
          />
        </div>

        <div className="gate__error" id="gate-error" role="alert">
          {error}
        </div>

        <button type="submit" className="btn btn--approve btn--full">
          Unlock console
        </button>

        <p className="gate__disclaimer">
          Non-authoritative research decision support. This console does not
          issue official tsunami products and cannot disseminate alerts.
        </p>
      </form>
    </div>
  );
}

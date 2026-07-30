import { Component } from "react";
import type { ReactNode, ErrorInfo } from "react";

interface Props {
  children: ReactNode;
  fallbackLabel?: string;
  /** Applied to the fallback element. The console is a CSS grid keyed on
   *  region classes, and html/body/#root are overflow: hidden, so a fallback
   *  that stands in for a grid-placed child without its class is auto-placed
   *  into an implicit row nobody can scroll to. */
  fallbackClassName?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Development only. The operator-facing message below is the production
    // channel; a deployed console has no reason to stream React component
    // stacks into a browser console it does not own.
    if (import.meta.env.DEV) {
      console.error(`[ErrorBoundary] ${this.props.fallbackLabel ?? "Panel"} crashed:`, error, info.componentStack);
    }
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          className={this.props.fallbackClassName}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            color: "var(--state-emergency)",
            fontSize: 11,
            gap: 8,
            padding: 16,
            textAlign: "center",
          }}
        >
          <span style={{ fontSize: 18 }}>&#9888;</span>
          <span style={{ fontWeight: 700, letterSpacing: "0.05em" }}>
            {this.props.fallbackLabel ?? "Panel"} Error
          </span>
          <span style={{ color: "var(--ink-dim)", fontSize: 10, maxWidth: 220 }}>
            {this.state.error?.message ?? "An unexpected error occurred."}
          </span>
          <button
            onClick={this.handleRetry}
            className="btn"
            style={{ marginTop: 4, padding: "5px 14px", fontSize: 10 }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

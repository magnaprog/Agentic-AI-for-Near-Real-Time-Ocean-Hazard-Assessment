import { Component } from "react";
import type { ReactNode, ErrorInfo } from "react";

interface Props {
  children: ReactNode;
  fallbackLabel?: string;
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
    console.error(`[ErrorBoundary] ${this.props.fallbackLabel ?? "Panel"} crashed:`, error, info.componentStack);
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
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

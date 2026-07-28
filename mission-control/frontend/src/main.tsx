import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Self-hosted fonts: an operations console must render identically without
// internet access, so the typefaces ship in the bundle instead of loading
// from a font CDN.
import "@fontsource/archivo/400.css";
import "@fontsource/archivo/600.css";
import "@fontsource/archivo/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/600.css";
import "@fontsource/ibm-plex-mono/700.css";
import "leaflet/dist/leaflet.css";
import App from "./App";
import "./styles/global.css";
import "./styles/dashboard.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);

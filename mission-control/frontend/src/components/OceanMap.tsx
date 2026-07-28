import { useEffect, memo } from "react";
import { MapContainer, TileLayer, Marker, Popup, Circle, Tooltip, useMap } from "react-leaflet";
import L from "leaflet";
import type { EventContext, StationRegistry } from "../types";
import { getEventModeStationIds } from "../constants";
import { MARKER_COLORS, halo } from "./mapMarkers";
import stationData from "../data/stations.json";

/** Names the map container for assistive technology.
 *
 *  Leaflet gives the container tabindex="0" for keyboard panning, so it is a
 *  tab stop; with no accessible name a screen reader announces the
 *  run-together text of every label inside it. react-leaflet v4 forwards only
 *  className, id and style to the div, so the attribute is set on the element
 *  Leaflet built.
 */
function MapAccessibleName() {
  const map = useMap();
  useEffect(() => {
    map
      .getContainer()
      .setAttribute("aria-label", "Pacific basin station map. Arrow keys pan, plus and minus zoom.");
  }, [map]);
  return null;
}

/** Pans the map when the target center changes. */
function ChangeView({ lat, lon, zoom }: { lat: number; lon: number; zoom: number }) {
  const map = useMap();
  useEffect(() => {
    map.setView([lat, lon], zoom);
  }, [map, lat, lon, zoom]);
  return null;
}

interface Props {
  eventContext: EventContext | null;
}

const stations: StationRegistry = stationData;

// Marker icons. Colors track the theme tokens in global.css. Each station
// marker is a luminous core with a soft halo so it reads against the dark
// basemap; the halo uses a translucent ring, not a heavy glow, so dense
// clusters do not smear together.
const epicenterIcon = new L.DivIcon({
  className: "",
  // The star glows in the emergency red rather than its own lighter fill, so
  // the epicenter reads at the same severity as the ESCALATE state color.
  html: `<div style="color:${MARKER_COLORS.epicenter};font-size:24px;font-weight:bold;text-align:center;line-height:1;text-shadow:0 0 6px rgba(224,92,92,0.7);">\u2605</div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

const dartIcon = new L.DivIcon({
  className: "",
  html: `<div style="width:8px;height:8px;border-radius:50%;background:${MARKER_COLORS.dart};box-shadow:0 0 0 2px ${halo(MARKER_COLORS.dart, 0.22)},0 0 6px ${halo(MARKER_COLORS.dart, 0.55)};"></div>`,
  iconSize: [8, 8],
  iconAnchor: [4, 4],
});

const dartActiveIcon = new L.DivIcon({
  className: "",
  html: `<div style="width:9px;height:9px;border-radius:50%;background:${MARKER_COLORS.dartActive};box-shadow:0 0 0 2px ${halo(MARKER_COLORS.dartActive, 0.25)},0 0 8px ${halo(MARKER_COLORS.dartActive, 0.7)};"></div>`,
  iconSize: [9, 9],
  iconAnchor: [4.5, 4.5],
});

const coopsIcon = new L.DivIcon({
  className: "",
  html: `<div style="width:5px;height:5px;border-radius:50%;background:${MARKER_COLORS.coops};box-shadow:0 0 0 1px ${halo(MARKER_COLORS.coops, 0.25)},0 0 4px ${halo(MARKER_COLORS.coops, 0.5)};"></div>`,
  iconSize: [5, 5],
  iconAnchor: [2.5, 2.5],
});

// Geographic labels rendered as markers to guarantee English text.
const geoLabels: { name: string; lat: number; lon: number; type: "continent" | "ocean" }[] = [
  { name: "North America", lat: 45, lon: -100, type: "continent" },
  { name: "South America", lat: -15, lon: -58, type: "continent" },
  { name: "Europe", lat: 50, lon: 15, type: "continent" },
  { name: "Africa", lat: 5, lon: 20, type: "continent" },
  { name: "Asia", lat: 45, lon: 90, type: "continent" },
  { name: "Oceania", lat: -25, lon: 135, type: "continent" },
  { name: "Antarctica", lat: -78, lon: 0, type: "continent" },
  { name: "Pacific Ocean", lat: 0, lon: -160, type: "ocean" },
  { name: "Atlantic Ocean", lat: 10, lon: -35, type: "ocean" },
  { name: "Indian Ocean", lat: -20, lon: 75, type: "ocean" },
  { name: "Arctic Ocean", lat: 78, lon: 0, type: "ocean" },
  { name: "Southern Ocean", lat: -62, lon: 0, type: "ocean" },
];

function geoLabelIcon(label: (typeof geoLabels)[number]) {
  const color = label.type === "continent" ? "rgba(190, 205, 220, 0.6)" : "rgba(110, 150, 190, 0.5)";
  const size = label.type === "continent" ? "11px" : "10px";
  return new L.DivIcon({
    className: "",
    html: `<div style="color:${color};font-size:${size};font-family:'Archivo',sans-serif;letter-spacing:2px;text-transform:uppercase;white-space:nowrap;pointer-events:none;text-shadow:0 0 4px rgba(0,0,0,0.85);">${label.name}</div>`,
    iconSize: [0, 0],
    iconAnchor: [0, 0],
  });
}

const geoLabelIcons = geoLabels.map((label) => ({
  ...label,
  icon: geoLabelIcon(label),
}));

/** Adjacent world copies to render each marker on. Leaflet wraps tile layers
 * across the antimeridian but not markers, which left the station set split at
 * +/-180 deg. Rendering each marker on the three neighboring copies keeps the
 * network complete no matter which copy is in view. minZoom is 3, where one
 * world is 2048 px wide; three copies are 6144 px, wider than any realistic
 * viewport, so these three always cover the visible copies. Lowering minZoom
 * would require more offsets. */
const WORLD_OFFSETS = [-360, 0, 360];

function OceanMap({ eventContext }: Props) {
  const center: [number, number] = eventContext
    ? [eventContext.epicenter_lat, eventContext.epicenter_lon]
    : [10, -170]; // Default: central Pacific

  const eventModeStations = getEventModeStationIds(eventContext);

  return (
    <div className="map-shell">
      <div className="map-vignette" aria-hidden />
      <MapContainer
        center={center}
        zoom={3}
        minZoom={3}
        style={{ height: "100%", width: "100%", background: "#070b11" }}
        zoomControl={false}
        worldCopyJump={true}
        attributionControl={false}
      >
        <MapAccessibleName />
        <ChangeView lat={center[0]} lon={center[1]} zoom={eventContext ? 5 : 3} />
        {/* Base: CARTO Dark (no labels) for neutral land silhouettes. */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
          attribution="&copy; OpenStreetMap contributors &copy; CARTO"
          maxZoom={13}
          className="base-tiles"
        />
        {/* Overlay: ESRI World Ocean Base bathymetry, multiply-blended so the
            ocean picks up depth shading while land stays neutral. {z}/{y}/{x}
            order is the ESRI convention. */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
          attribution="Tiles &copy; Esri - Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri"
          maxZoom={13}
          className="ocean-overlay"
        />

        {/* keyboard={false} on every marker below. Leaflet's Marker default
            is keyboard: true, and leaflet-src.js:7915-7916 then sets
            tabIndex="0" and role="button" on each icon. That guard reads
            options.keyboard, NOT options.interactive, so interactive={false}
            did not stop it: 39 DART + 81 CO-OPS + 12 labels across three world
            copies put ~400 unlabelled buttons in the tab order ahead of the
            review gate (measured: 402 tabs to reach ACKNOWLEDGE). The map is a
            reference visualisation and every datum it carries is also rendered
            as text in the side panels, so it is excluded from the tab order
            rather than given 400 accessible names. Do not restore keyboard
            access without also moving the map out of the path to the decision
            controls. */}
        {geoLabelIcons.map((g) =>
          WORLD_OFFSETS.map((off) => (
            <Marker key={`${g.name}-${off}`} position={[g.lat, g.lon + off]} icon={g.icon} interactive={false} keyboard={false} />
          ))
        )}

        {eventContext && eventContext.seismic_region !== "" && (
          <>
            {WORLD_OFFSETS.map((off) => (
              <Marker
                key={`epi-${off}`}
                position={[eventContext.epicenter_lat, eventContext.epicenter_lon + off]}
                icon={epicenterIcon}
                keyboard={false}
              >
                <Popup>
                  <strong>Epicenter</strong>
                  <br />
                  M{eventContext.seismic_magnitude}: {eventContext.seismic_region}
                  <br />
                  {eventContext.epicenter_lat.toFixed(2)}, {eventContext.epicenter_lon.toFixed(2)}
                </Popup>
              </Marker>
            ))}
            {WORLD_OFFSETS.map((off) => (
              <Circle
                key={`epi-circle-${off}`}
                center={[eventContext.epicenter_lat, eventContext.epicenter_lon + off]}
                radius={500000}
                pathOptions={{
                  color: "#e05c5c",
                  fillColor: "#e05c5c",
                  fillOpacity: 0.06,
                  weight: 1,
                  dashArray: "4 4",
                }}
              />
            ))}
          </>
        )}

        {/* Reference DART inventory; event-mode evidence is highlighted. */}
        {stations.dart.map((s) =>
          WORLD_OFFSETS.map((off) => (
            <Marker
              key={`dart-${s.id}-${off}`}
              position={[s.lat, s.lon + off]}
              icon={eventModeStations.has(s.id) ? dartActiveIcon : dartIcon}
              keyboard={false}
            >
              {eventModeStations.has(s.id) && (
                <Tooltip permanent direction="right" offset={[6, 0]} className="station-label">
                  {s.id}
                </Tooltip>
              )}
              <Popup>
                <strong>{s.name}</strong>
                <br />
                {s.lat.toFixed(2)}, {s.lon.toFixed(2)}
                <br />
                {eventModeStations.has(s.id) ? "Event mode observed" : "Reference inventory"}
              </Popup>
            </Marker>
          ))
        )}

        {/* Reference CO-OPS inventory; live activity is not exposed here. */}
        {stations.coops.map((s) =>
          WORLD_OFFSETS.map((off) => (
            <Marker
              key={`coops-${s.id}-${off}`}
              position={[s.lat, s.lon + off]}
              icon={coopsIcon}
              keyboard={false}
            >
              <Popup>
                <strong>{s.name}</strong>
                <br />
                {s.lat.toFixed(2)}, {s.lon.toFixed(2)}
              </Popup>
            </Marker>
          ))
        )}
      </MapContainer>

      {/* Leaflet's attribution control is disabled to keep the dark theme, so
          the tile providers' required attribution is rendered here instead. */}
      <div className="map-attribution">
        &copy; OpenStreetMap contributors &copy; CARTO. Ocean bathymetry tiles &copy; Esri,
        sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ.
      </div>
      <div className="map-legend">
        <div className="map-legend__title">Legend</div>
        <div className="map-legend__row">
          <span className="map-legend__swatch" style={{ width: 8, height: 8, borderRadius: "50%", background: MARKER_COLORS.dart }} />
          DART reference location
        </div>
        <div className="map-legend__row">
          <span className="map-legend__swatch" style={{ width: 8, height: 8, borderRadius: "50%", background: MARKER_COLORS.dartActive }} />
          DART event mode observed
        </div>
        <div className="map-legend__row">
          <span className="map-legend__swatch" style={{ width: 6, height: 6, borderRadius: "50%", background: MARKER_COLORS.coops }} />
          Coastal gauge
        </div>
        <div className="map-legend__row">
          <span className="map-legend__swatch" style={{ color: MARKER_COLORS.epicenter, fontSize: 13, lineHeight: 1, width: 8, textAlign: "center" }}>
            {"\u2605"}
          </span>
          Epicenter
        </div>
      </div>
    </div>
  );
}

export default memo(OceanMap);

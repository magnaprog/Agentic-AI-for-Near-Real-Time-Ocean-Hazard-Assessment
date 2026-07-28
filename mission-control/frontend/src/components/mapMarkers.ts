/** Marker palette shared by the map DivIcons and the legend swatches, so a
 * recolor cannot leave the legend behind. Lives in its own module (not the
 * OceanMap component file) so exporting it does not mix a constant with a
 * component export, which react-refresh flags. */

export const MARKER_COLORS = {
  dart: "#3ee0c8",
  dartActive: "#ff6b6b",
  coops: "#f0b86a",
  epicenter: "#ff7b7b",
} as const;

/** Same color at an alpha, for the marker halos. */
export function halo(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

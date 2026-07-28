import { useRef, useEffect, useState, useCallback, useId, memo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { Thresholds } from "../types";
import { DEFAULT_THRESHOLDS } from "../constants";

interface Props {
  currentScore: number;
  thresholds: Thresholds | null;
  /** When false there is no active event, so no live score is being produced
   * and the chart shows an idle message instead of a flat zero line. */
  hasActiveEvent: boolean;
}

interface DataPoint {
  /** Epoch milliseconds. The x-axis is a real time scale, not a category
   *  axis: with a string dataKey Recharts spaces points evenly regardless of
   *  the interval between them, so a 2-second gap and the 30-second keepalive
   *  gap drew the same width and any slope read off this chart was a
   *  distorted rate. */
  t: number;
  time: string;
  score: number;
}

const MAX_POINTS = 60;

// Recharts renders SVG, so CSS variables do not resolve here; hexes track the
// theme tokens in global.css.
const COLOR_MONITOR = "#e0a45c";
const COLOR_ACCENT = "#46c2ae";
const COLOR_EMERGENCY = "#e05c5c";
const COLOR_LINE = "#3aa87f";
const COLOR_FILL = "#3aa87f";
const COLOR_AXIS = "#223140";
const COLOR_TICK = "#6a7c8d";
const FONT_MONO = "'IBM Plex Mono', monospace";

function AnomalyChart({ currentScore, thresholds, hasActiveEvent }: Props) {
  const gradientId = useId();
  const [data, setData] = useState<DataPoint[]>([]);
  const scoreRef = useRef(currentScore);
  const lastAppendMsRef = useRef(0);

  const appendPoint = useCallback((score: number, nowMs: number = Date.now()) => {
    const now = new Date(nowMs).toISOString().slice(11, 19);
    setData((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.time === now && last.score === score) {
        return prev;
      }
      const next = [...prev, { t: nowMs, time: now, score }];
      return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
    });
    lastAppendMsRef.current = nowMs;
  }, []);

  useEffect(() => {
    scoreRef.current = currentScore;
  }, [currentScore]);

  // Reset the trace when an event begins or ends so a stale event's line does
  // not bleed into the next event's view.
  useEffect(() => {
    setData([]);
    lastAppendMsRef.current = 0;
  }, [hasActiveEvent]);

  useEffect(() => {
    if (hasActiveEvent) appendPoint(currentScore);
  }, [currentScore, hasActiveEvent]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!hasActiveEvent) return;
    const id = setInterval(() => {
      const nowMs = Date.now();
      if (nowMs - lastAppendMsRef.current >= 30_000) {
        appendPoint(scoreRef.current, nowMs);
      }
    }, 30_000);
    return () => clearInterval(id);
  }, [hasActiveEvent]); // eslint-disable-line react-hooks/exhaustive-deps

  const t1 = thresholds?.t1 ?? DEFAULT_THRESHOLDS.t1;
  const t2 = thresholds?.t2 ?? DEFAULT_THRESHOLDS.t2;
  const t3 = thresholds?.t3 ?? DEFAULT_THRESHOLDS.t3;

  if (!hasActiveEvent) {
    return (
      <div className="standby" style={{ height: "100%" }}>
        <div className="standby__mark">NO ACTIVE EVENT</div>
        <div className="standby__text">Live anomaly scoring resumes when an event is triggered.</div>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 2, left: -10 }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLOR_FILL} stopOpacity={0.12} />
            <stop offset="95%" stopColor={COLOR_FILL} stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis
          dataKey="t"
          type="number"
          scale="time"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(ms: number) => new Date(ms).toISOString().slice(11, 19)}
          tick={{ fontSize: 9, fill: COLOR_TICK, fontFamily: FONT_MONO }}
          stroke={COLOR_AXIS}
          tickMargin={4}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={[0, 1]}
          ticks={[0, t1, t2, t3, 1]}
          tick={{ fontSize: 9, fill: COLOR_TICK, fontFamily: FONT_MONO }}
          stroke={COLOR_AXIS}
          width={34}
        />
        <Tooltip
          contentStyle={{
            background: "#0a0f16",
            border: "1px solid #223140",
            borderRadius: 3,
            fontSize: 11,
            color: "#d9e2ec",
            fontFamily: FONT_MONO,
          }}
          itemStyle={{ color: "#d9e2ec" }}
          labelFormatter={(ms) => `${new Date(Number(ms)).toISOString().slice(11, 19)} UTC`}
        />
        <ReferenceLine
          y={t1}
          stroke={COLOR_MONITOR}
          strokeDasharray="6 4"
          strokeOpacity={0.6}
          label={{ value: `T1=${t1}`, position: "insideTopRight", fill: COLOR_MONITOR, fontSize: 9, fontWeight: 700, fontFamily: FONT_MONO }}
        />
        <ReferenceLine
          y={t2}
          stroke={COLOR_ACCENT}
          strokeDasharray="6 4"
          strokeOpacity={0.5}
          label={{ value: `T2=${t2}`, position: "insideTopRight", fill: COLOR_ACCENT, fontSize: 9, fontWeight: 700, fontFamily: FONT_MONO }}
        />
        <ReferenceLine
          y={t3}
          stroke={COLOR_EMERGENCY}
          strokeDasharray="6 4"
          strokeOpacity={0.7}
          label={{ value: `T3=${t3}`, position: "insideTopRight", fill: COLOR_EMERGENCY, fontSize: 9, fontWeight: 700, fontFamily: FONT_MONO }}
        />
        <Area
          type="monotone"
          dataKey="score"
          stroke={COLOR_LINE}
          strokeWidth={2}
          fillOpacity={1}
          fill={`url(#${gradientId})`}
          activeDot={{ r: 3, fill: COLOR_LINE, stroke: "#fff", strokeWidth: 1 }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default memo(AnomalyChart);

import { useEffect, useMemo, useState } from "react";
import {
  Activity, Check, ChevronDown, ChevronUp, ClipboardCheck, Coins, GitBranch,
  Loader2, MessageSquare, ShieldAlert, ThumbsDown, ThumbsUp, Timer,
} from "lucide-react";
import { monitoringAPI } from "../api/client";
import type { DownvotedItem, MonitoringStats } from "../api/client";
import { useI18n, type TFunc } from "../i18n/LanguageProvider";

// Single data hue (validated vs light surface — see dataviz palette check)
const DATA_HUE = "#7C3AED";
const INK = "#1A1A2E";
const MUTED = "#9CA3AF";

const COMPLEXITY_KEYS = ["simple", "medium", "complex", "multihop"] as const;
const complexityLabel = (t: TFunc, key: string): string =>
  (COMPLEXITY_KEYS as readonly string[]).includes(key)
    ? t(`dash.complexity.${key}` as `dash.complexity.simple`)
    : key;

function StatTile({ icon, label, value, sub }: {
  icon: React.ReactNode; label: string; value: string; sub?: string;
}) {
  return (
    <div
      className="flex-1 min-w-[150px] rounded-2xl px-4 py-3.5"
      style={{ background: "rgba(255,255,255,0.65)", border: "1px solid rgba(124,58,237,0.12)" }}
    >
      <div className="flex items-center gap-2 text-[11px] font-medium" style={{ color: MUTED }}>
        <span style={{ color: DATA_HUE }}>{icon}</span>
        {label}
      </div>
      <p className="text-xl font-bold mt-1.5" style={{ color: INK }}>{value}</p>
      {sub && <p className="text-[11px] mt-0.5" style={{ color: MUTED }}>{sub}</p>}
    </div>
  );
}

/** Bar chart: queries per day. Single series (one hue), thin bars with
 *  rounded data-ends, 2px gaps, recessive grid, per-bar hover tooltip. */
function QueriesPerDayChart({ data }: { data: MonitoringStats["per_day"] }) {
  const { t } = useI18n();
  const [hover, setHover] = useState<number | null>(null);
  const W = 640, H = 180, PAD_L = 30, PAD_B = 22, PAD_T = 12;
  const max = Math.max(1, ...data.map((d) => d.queries));
  const innerW = W - PAD_L - 8;
  const innerH = H - PAD_T - PAD_B;
  const slot = innerW / data.length;
  const barW = Math.min(22, Math.max(6, slot - 2)); // ≥2px gap between bars
  const maxIdx = data.reduce((mi, d, i) => (d.queries > data[mi].queries ? i : mi), 0);
  const gridVals = [Math.round(max / 2), max];

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
        aria-label={t("dash.queriesPerDayAria")}>
        {/* recessive gridlines + muted y labels */}
        {gridVals.map((v) => {
          const y = PAD_T + innerH * (1 - v / max);
          return (
            <g key={v}>
              <line x1={PAD_L} x2={W - 8} y1={y} y2={y}
                stroke="rgba(26,26,46,0.08)" strokeWidth="1" />
              <text x={PAD_L - 6} y={y + 3} textAnchor="end" fontSize="9" fill={MUTED}>{v}</text>
            </g>
          );
        })}
        {/* baseline */}
        <line x1={PAD_L} x2={W - 8} y1={PAD_T + innerH} y2={PAD_T + innerH}
          stroke="rgba(26,26,46,0.18)" strokeWidth="1" />

        {data.map((d, i) => {
          const h = d.queries === 0 ? 0 : Math.max(3, innerH * (d.queries / max));
          const x = PAD_L + i * slot + (slot - barW) / 2;
          const y = PAD_T + innerH - h;
          const dayLabel = d.date.slice(8, 10) + "/" + d.date.slice(5, 7);
          return (
            <g key={d.date}>
              {d.queries > 0 && (
                <rect x={x} y={y} width={barW} height={h} rx={4} ry={4}
                  fill={DATA_HUE} opacity={hover === null || hover === i ? 1 : 0.45} />
              )}
              {/* rounded top only: square off the bottom against the baseline */}
              {d.queries > 0 && h > 4 && (
                <rect x={x} y={PAD_T + innerH - 4} width={barW} height={4}
                  fill={DATA_HUE} opacity={hover === null || hover === i ? 1 : 0.45} />
              )}
              {/* selective direct label: max day only */}
              {i === maxIdx && d.queries > 0 && (
                <text x={x + barW / 2} y={y - 4} textAnchor="middle" fontSize="9"
                  fontWeight="600" fill={INK}>{d.queries}</text>
              )}
              {/* x labels: every other day to avoid collisions */}
              {i % 2 === 0 && (
                <text x={x + barW / 2} y={H - 8} textAnchor="middle" fontSize="8.5"
                  fill={MUTED}>{dayLabel}</text>
              )}
              {/* hover hit target wider than the mark */}
              <rect x={PAD_L + i * slot} y={PAD_T} width={slot} height={innerH}
                fill="transparent"
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)} />
            </g>
          );
        })}
      </svg>
      {hover !== null && (
        <div
          className="absolute pointer-events-none rounded-lg px-2.5 py-1.5 text-[11px] shadow-md"
          style={{
            left: `${((PAD_L + hover * slot + slot / 2) / W) * 100}%`,
            top: 0,
            transform: "translateX(-50%)",
            background: "rgba(26,26,46,0.92)",
            color: "#fff",
            whiteSpace: "nowrap",
          }}
        >
          {data[hover].date} · {data[hover].queries} {t("dash.queriesUnit")}
          {data[hover].multihop > 0 && ` (${data[hover].multihop} ${t("dash.multihopUnit")})`}
        </div>
      )}
    </div>
  );
}

/** Horizontal magnitude bars per complexity tier — identity lives in the
 *  row label, so a single hue is correct here. */
function ComplexityBreakdown({ data }: { data: Record<string, number> }) {
  const { t } = useI18n();
  const rows = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...rows.map(([, v]) => v));
  if (rows.length === 0) {
    return <p className="text-xs" style={{ color: MUTED }}>{t("dash.noData")}</p>;
  }
  return (
    <div className="space-y-2">
      {rows.map(([key, count]) => (
        <div key={key} className="flex items-center gap-3 text-xs">
          <span className="w-24 shrink-0 text-right" style={{ color: INK }}>
            {complexityLabel(t, key)}
          </span>
          <div className="flex-1 h-4 rounded-md overflow-hidden" style={{ background: "rgba(124,58,237,0.07)" }}>
            <div
              className="h-full rounded-md"
              style={{ width: `${(count / max) * 100}%`, background: DATA_HUE, minWidth: count > 0 ? 4 : 0 }}
              title={`${count} ${t("dash.queriesUnit")}`}
            />
          </div>
          <span className="w-10 shrink-0 font-semibold" style={{ color: INK }}>{count}</span>
        </div>
      ))}
    </div>
  );
}

/** Downvoted answers → the online-feedback-to-offline-eval loop. Each row can
 *  be pushed into the eval queue (data/feedback_eval_queue.json) for later
 *  ground-truth labelling and regression tracking (TASKLIST D4). */
function DownvotedPanel({ days }: { days: number }) {
  const { t } = useI18n();
  const [items, setItems] = useState<DownvotedItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [queuing, setQueuing] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setItems(null);
    setError(null);
    monitoringAPI.downvoted(Math.max(days, 30))
      .then((d) => { if (!cancelled) setItems(d.items); })
      .catch(() => { if (!cancelled) setError(t("dash.dv.loadError")); });
    return () => { cancelled = true; };
  }, [days]);

  const addToQueue = async (id: string) => {
    setQueuing(id);
    try {
      await monitoringAPI.addToEvalQueue(id);
      setItems((prev) => prev?.map((i) =>
        i.message_id === id ? { ...i, in_eval_queue: true } : i) ?? null);
    } catch { /* keep button state; user can retry */ }
    finally { setQueuing(null); }
  };

  return (
    <div
      className="rounded-2xl px-5 py-4 mb-10"
      style={{ background: "rgba(255,255,255,0.65)", border: "1px solid rgba(124,58,237,0.12)" }}
    >
      <p className="text-sm font-semibold mb-1 flex items-center gap-2" style={{ color: INK }}>
        <ThumbsDown className="w-4 h-4" style={{ color: "#EF4444" }} />
        {t("dash.dv.title")}
      </p>
      <p className="text-[11px] mb-3" style={{ color: MUTED }}>
        {t("dash.dv.note")}
      </p>

      {!items && !error && (
        <div className="flex items-center gap-2 py-4 text-xs" style={{ color: MUTED }}>
          <Loader2 className="w-4 h-4 animate-spin" /> {t("common.loading")}
        </div>
      )}
      {error && <p className="text-xs py-3" style={{ color: "#B91C1C" }}>{error}</p>}
      {items && items.length === 0 && (
        <p className="text-xs py-3 text-center" style={{ color: MUTED }}>
          {t("dash.dv.empty")}
        </p>
      )}

      <div className="space-y-2">
        {items?.map((it) => (
          <div key={it.message_id} className="rounded-xl overflow-hidden"
            style={{ background: "rgba(255,255,255,0.7)", border: "1px solid rgba(239,68,68,0.15)" }}>
            <div className="flex items-start gap-2 px-3 py-2.5">
              <button
                onClick={() => setExpanded(expanded === it.message_id ? null : it.message_id)}
                className="min-w-0 flex-1 text-left"
              >
                <p className="text-xs font-semibold truncate" style={{ color: INK }}>
                  {it.question || t("dash.dv.unknownQuestion")}
                </p>
                <p className="text-[11px] mt-0.5 line-clamp-1" style={{ color: MUTED }}>
                  {it.answer}
                </p>
              </button>
              {it.in_eval_queue ? (
                <span className="shrink-0 h-7 px-2 rounded-lg flex items-center gap-1 text-[11px] font-medium"
                  style={{ background: "rgba(16,185,129,0.1)", color: "#059669" }}>
                  <Check className="w-3.5 h-3.5" /> {t("dash.dv.added")}
                </span>
              ) : (
                <button
                  onClick={() => addToQueue(it.message_id)}
                  disabled={queuing === it.message_id}
                  title={t("dash.dv.addTitle")}
                  className="shrink-0 h-7 px-2.5 rounded-lg flex items-center gap-1 text-[11px] font-semibold text-white transition-colors disabled:opacity-50"
                  style={{ background: DATA_HUE }}
                >
                  {queuing === it.message_id
                    ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    : <ClipboardCheck className="w-3.5 h-3.5" />}
                  {t("dash.dv.add")}
                </button>
              )}
              <button
                onClick={() => setExpanded(expanded === it.message_id ? null : it.message_id)}
                className="shrink-0 w-7 h-7 rounded-lg flex items-center justify-center text-gray-400 hover:bg-black/5"
              >
                {expanded === it.message_id ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>
            {expanded === it.message_id && (
              <div className="px-3 pb-3 pt-1 text-[11px] whitespace-pre-wrap leading-relaxed"
                style={{ color: "#374151", borderTop: "1px solid rgba(239,68,68,0.1)" }}>
                {it.answer}
                {it.sources.length > 0 && (
                  <p className="mt-2" style={{ color: MUTED }}>
                    {t("dash.dv.sources", { list: it.sources.map((s) => s.title).join(" · ") })}
                  </p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { t } = useI18n();
  const [stats, setStats] = useState<MonitoringStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState(14);

  useEffect(() => {
    let cancelled = false;
    setStats(null);
    setError(null);
    monitoringAPI.stats(days)
      .then((d) => { if (!cancelled) setStats(d); })
      .catch((e) => {
        if (!cancelled) {
          setError(e?.response?.status === 403
            ? t("dash.errAdmin")
            : t("dash.errLoad"));
        }
      });
    return () => { cancelled = true; };
  }, [days, t]);

  const fmtLat = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`);
  const kpis = useMemo(() => {
    if (!stats) return [];
    return [
      {
        icon: <MessageSquare className="w-3.5 h-3.5" />, label: t("dash.kpi.queries"),
        value: String(stats.n_queries),
        sub: t("dash.kpi.queriesSub", { multihop: stats.multihop_queries, errors: stats.errors }),
      },
      {
        icon: <Timer className="w-3.5 h-3.5" />, label: t("dash.kpi.latency"),
        value: fmtLat(stats.latency_ms.p50),
        sub: t("dash.kpi.latencySub", { p95: fmtLat(stats.latency_ms.p95), p99: fmtLat(stats.latency_ms.p99) }),
      },
      {
        icon: <Coins className="w-3.5 h-3.5" />, label: t("dash.kpi.cost"),
        value: `$${stats.total_cost_usd.toFixed(4)}`,
        sub: t("dash.kpi.costSub", { tokens: stats.total_tokens.toLocaleString() }),
      },
      {
        icon: <Activity className="w-3.5 h-3.5" />, label: t("dash.kpi.errorRate"),
        value: `${(stats.error_rate * 100).toFixed(1)}%`,
        sub: t("dash.kpi.errorRateSub", { aborted: stats.aborted }),
      },
    ];
  }, [stats, t]);

  return (
    <div
      className="flex flex-col flex-1 h-full overflow-y-auto px-4 py-5 sm:px-6 md:px-10 md:py-8 pt-16 md:pt-8"
      style={{
        background: "rgba(255,255,255,0.3)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
      }}
    >
      <div className="flex flex-wrap items-start justify-between gap-3 sm:gap-4 mb-5 md:mb-6">
        <div className="flex items-center gap-3">
          <div
            className="w-11 h-11 rounded-2xl flex items-center justify-center"
            style={{ background: "rgba(124,58,237,0.12)", color: DATA_HUE }}
          >
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold" style={{ color: INK }}>{t("dash.title")}</h1>
            <p className="text-xs mt-0.5" style={{ color: MUTED }}>
              {t("dash.subtitle", { days })}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {[7, 14, 30].map((d) => (
            <button
              key={d}
              onClick={() => setDays(d)}
              className="h-8 px-3 rounded-xl text-xs font-medium transition-colors"
              style={days === d
                ? { background: DATA_HUE, color: "#fff" }
                : { background: "rgba(124,58,237,0.07)", color: DATA_HUE }}
            >
              {t("dash.days", { n: d })}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div
          className="flex items-center gap-2.5 rounded-2xl px-4 py-3 text-sm"
          style={{ background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.2)", color: "#B91C1C" }}
        >
          <ShieldAlert className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {!error && !stats && (
        <div className="flex items-center justify-center gap-2 mt-16 text-sm" style={{ color: MUTED }}>
          <Loader2 className="w-4 h-4 animate-spin" /> {t("dash.loading")}
        </div>
      )}

      {stats && (
        <>
          {/* KPI row */}
          <div className="flex flex-wrap gap-3 mb-6">
            {kpis.map((k) => <StatTile key={k.label} {...k} />)}
          </div>

          {/* Queries per day */}
          <div
            className="rounded-2xl px-5 py-4 mb-5"
            style={{ background: "rgba(255,255,255,0.65)", border: "1px solid rgba(124,58,237,0.12)" }}
          >
            <p className="text-sm font-semibold mb-3" style={{ color: INK }}>
              {t("dash.queriesPerDay")}
            </p>
            {stats.n_queries === 0 ? (
              <p className="text-xs py-6 text-center" style={{ color: MUTED }}>
                {t("dash.noQueries")}
              </p>
            ) : (
              <QueriesPerDayChart data={stats.per_day} />
            )}
          </div>

          <div className="flex flex-wrap gap-5 pb-10">
            {/* Complexity breakdown */}
            <div
              className="flex-1 min-w-[280px] rounded-2xl px-5 py-4"
              style={{ background: "rgba(255,255,255,0.65)", border: "1px solid rgba(124,58,237,0.12)" }}
            >
              <p className="text-sm font-semibold mb-3 flex items-center gap-2" style={{ color: INK }}>
                <GitBranch className="w-4 h-4" style={{ color: DATA_HUE }} />
                {t("dash.complexityTitle")}
              </p>
              <ComplexityBreakdown data={stats.by_complexity} />
            </div>

            {/* Feedback */}
            <div
              className="flex-1 min-w-[220px] rounded-2xl px-5 py-4"
              style={{ background: "rgba(255,255,255,0.65)", border: "1px solid rgba(124,58,237,0.12)" }}
            >
              <p className="text-sm font-semibold mb-3" style={{ color: INK }}>
                {t("dash.feedbackTitle")}
              </p>
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <ThumbsUp className="w-4 h-4" style={{ color: "#10B981" }} />
                  <span className="text-lg font-bold" style={{ color: INK }}>{stats.feedback.up}</span>
                  <span className="text-xs" style={{ color: MUTED }}>{t("dash.feedbackUp")}</span>
                </div>
                <div className="flex items-center gap-2">
                  <ThumbsDown className="w-4 h-4" style={{ color: "#EF4444" }} />
                  <span className="text-lg font-bold" style={{ color: INK }}>{stats.feedback.down}</span>
                  <span className="text-xs" style={{ color: MUTED }}>{t("dash.feedbackDown")}</span>
                </div>
              </div>
              <p className="text-[11px] mt-3" style={{ color: MUTED }}>
                {t("dash.feedbackNote")}
              </p>
            </div>
          </div>

          {/* Downvoted answers → eval queue (feedback loop) */}
          <DownvotedPanel days={days} />
        </>
      )}
    </div>
  );
}

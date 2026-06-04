import type { ReportSource, ReportStock } from "@/lib/ai-report/types";

export function getPredictHref(stock: ReportStock) {
  const recommendationReasons = [
    stock.reasons.news.summary,
    stock.reasons.news.details,
    stock.reasons.disclosure.summary,
    stock.reasons.disclosure.details,
    stock.reasons.flow.summary,
    stock.reasons.flow.details,
  ]
    .filter(Boolean)
    .join(" ")
    .slice(0, 1200);

  const params = new URLSearchParams({
    code: stock.code,
    name: stock.name,
    price: stock.price,
    change: stock.change,
    recommendationSignal: stock.signal,
    recommendationSummary: stock.summary.slice(0, 500),
    recommendationReasons,
  });

  return `/ai-report/predict?${params.toString()}`;
}

export function getSourceHref(source: ReportSource) {
  const href =
    typeof source === "string"
      ? source.trim()
      : (source.url || source.href || "").trim();
  if (/^https?:\/\//i.test(href)) return href;
  if (/^www\./i.test(href)) return `https://${href}`;
  return null;
}

export function getSourceLabel(source: ReportSource) {
  if (typeof source === "string") return source;
  return source.label || source.title || source.url || source.href || "출처";
}

export function getSignalBadgeClass(signal: string) {
  const normalized = signal.toLowerCase();

  if (signal.includes("매수") || normalized.includes("buy")) {
    return "bg-rose-50 text-rose-600";
  }
  if (signal.includes("매도") || normalized.includes("sell")) {
    return "bg-blue-50 text-blue-600";
  }
  if (signal.includes("관망") || normalized.includes("hold")) {
    return "bg-slate-100 text-slate-500";
  }

  return "bg-slate-100 text-slate-500";
}

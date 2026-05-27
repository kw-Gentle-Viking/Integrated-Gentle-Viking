export type ReasonKey = "news" | "disclosure" | "flow";

export type ReportSource =
  | string
  | {
      label?: string;
      title?: string;
      url?: string;
      href?: string;
    };

export type GeminiReason = {
  summary: string;
  details: string;
  sources: ReportSource[];
};

export type ReportStock = {
  rank: number;
  name: string;
  code: string;
  logoText: string;
  price: string;
  change: string;
  signal: string;
  score: number;
  summary: string;
  reasons: Record<ReasonKey, GeminiReason>;
};

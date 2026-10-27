"use client";

// 종목 상세 페이지를 열 때마다 조회한 종목코드를 이 브라우저에만 남겨둔다(서버 저장 없음).
// 대시보드 "최근에 본 종목" 카드가 이 목록을 읽어 실시간 시세와 함께 보여준다.
const STORAGE_KEY = "gv-recent-stocks";
const MAX_RECENT = 8;

export function recordRecentStock(code: string) {
  if (typeof window === "undefined" || !code) return;
  try {
    const existing = getRecentStockCodes();
    const next = [code, ...existing.filter((c) => c !== code)].slice(0, MAX_RECENT);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // localStorage 접근 불가(프라이빗 모드 등) -- 조회 이력 저장만 건너뛴다
  }
}

export function getRecentStockCodes(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list.filter((c): c is string => typeof c === "string") : [];
  } catch {
    return [];
  }
}

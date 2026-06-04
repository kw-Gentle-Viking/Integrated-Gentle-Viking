"use client";

import { useEffect, useSyncExternalStore } from "react";
import { apiFetch } from "@/lib/signup/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type StockHolding = {
  flag: string;
  label: string;
  valueKRW: number;
  pnlKRW: number;
  pnlRate: number;
};

export type MonthlyReturn = {
  totalKRW: number;
  saleKRW: number;
  dividendKRW: number;
  interestKRW: number;
};

export type AccountAssets = {
  broker: string;
  accountNo: string;
  totalKRW: number;
  orderableKRW: number;
  cashKRW: number;
  investedKRW: number;
  investedPnlKRW: number;
  investedPnlRate: number;
  holdings: StockHolding[];
  monthly: MonthlyReturn;
};

type AssetsSnapshot = {
  assets: AccountAssets | null;
  isLoading: boolean;
  isRefreshing: boolean;
  error: string | null;
  lastLoadedAt: number | null;
};

const emptyMonthly: MonthlyReturn = {
  totalKRW: 0,
  saleKRW: 0,
  dividendKRW: 0,
  interestKRW: 0,
};

let snapshot: AssetsSnapshot = {
  assets: null,
  isLoading: true,
  isRefreshing: false,
  error: null,
  lastLoadedAt: null,
};
let currentRequest: Promise<AccountAssets | null> | null = null;
const listeners = new Set<() => void>();

function normalizeAssets(payload: Partial<AccountAssets>): AccountAssets {
  return {
    broker: payload.broker ?? "한국투자증권",
    accountNo: payload.accountNo ?? "-",
    totalKRW: Number(payload.totalKRW ?? 0),
    orderableKRW: Number(payload.orderableKRW ?? 0),
    cashKRW: Number(payload.cashKRW ?? 0),
    investedKRW: Number(payload.investedKRW ?? 0),
    investedPnlKRW: Number(payload.investedPnlKRW ?? 0),
    investedPnlRate: Number(payload.investedPnlRate ?? 0),
    holdings: payload.holdings ?? [],
    monthly: payload.monthly ?? emptyMonthly,
  };
}

function fingerprintAssets(assets: AccountAssets | null) {
  return assets ? JSON.stringify(assets) : "";
}

function emit() {
  listeners.forEach((listener) => listener());
}

function setSnapshot(next: Partial<AssetsSnapshot>) {
  snapshot = { ...snapshot, ...next };
  emit();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot() {
  return snapshot;
}

export async function refreshAccountAssets(options: { force?: boolean } = {}) {
  if (currentRequest) return currentRequest;
  if (snapshot.assets && !options.force) return Promise.resolve(snapshot.assets);

  setSnapshot({
    isLoading: !snapshot.assets,
    isRefreshing: Boolean(snapshot.assets),
    error: null,
  });

  currentRequest = (async () => {
    try {
      const res = await apiFetch(`${API_BASE}/account/assets`, { cache: "no-store" });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(payload.detail || payload.message || `자산 조회 실패 (${res.status})`);
      }

      const nextAssets = normalizeAssets(payload as Partial<AccountAssets>);
      const changed = fingerprintAssets(nextAssets) !== fingerprintAssets(snapshot.assets);
      setSnapshot({
        assets: changed || !snapshot.assets ? nextAssets : snapshot.assets,
        isLoading: false,
        isRefreshing: false,
        error: null,
        lastLoadedAt: Date.now(),
      });
      return nextAssets;
    } catch (error) {
      setSnapshot({
        isLoading: false,
        isRefreshing: false,
        error: error instanceof Error ? error.message : "자산 정보를 불러오지 못했습니다.",
      });
      return snapshot.assets;
    } finally {
      currentRequest = null;
    }
  })();

  return currentRequest;
}

export function invalidateAccountAssets() {
  void refreshAccountAssets({ force: true });
}

export function useAccountAssets() {
  const store = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  useEffect(() => {
    void refreshAccountAssets();
  }, []);

  return store;
}

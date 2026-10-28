"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { AUTH_EVENT_NAME, apiFetch, getAccessToken } from "@/lib/signup/auth";
import { fetchCurrentPrice } from "@/lib/api/prices";

export type StockListItem = {
  code: string;
  name: string;
  price: string;
  change: string;
  logoText?: string;
  memo?: string;
};

interface StockListContextType {
  favorites: string[];
  cart: string[];
  favoriteStocks: StockListItem[];
  cartStocks: StockListItem[];
  toggleFavorite: (code: string, stock?: StockListItem) => void;
  toggleCart: (code: string, stock?: StockListItem) => void;
  isFavorite: (code: string) => boolean;
  isInCart: (code: string) => boolean;
}

const MAX_CART_STOCKS = 5;

const StockListContext = createContext<StockListContextType | undefined>(undefined);

// 바구니(basket) 저장값(ticker_name)은 담을 당시 클라이언트가 보낸 값을 그대로 echo하는
// 컬럼이라 테스트 데이터 등으로 오염될 수 있다. KIS 현재가 조회(`/prices/current`)는
// 종목명을 내려주지 않으므로(inquire-price 응답엔 hts_kor_isnm이 없음, 업종명만 있음),
// 지금 AI가 실제로 추론하는 종목만이라도 정확한 이름으로 보정한다. 여기 없는 종목은
// 바구니 저장값 -> 없으면 종목코드 순으로 fallback.
const KNOWN_STOCK_NAMES: Record<string, string> = {
  "005930": "삼성전자",
  "000660": "SK하이닉스",
  "005380": "현대차",
  "068270": "셀트리온",
  "207940": "삼성바이오로직스",
};

export function resolveStockName(code: string, fallback: string): string {
  return KNOWN_STOCK_NAMES[code] ?? fallback;
}

type BackendBasketItem = {
  ticker: string;
  ticker_name: string;
};

async function addBasketItem(stock: StockListItem) {
  await apiFetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/basket`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker: stock.code, ticker_name: stock.name }),
  });
}

async function removeBasketItem(code: string) {
  await apiFetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/basket/${encodeURIComponent(code)}`, {
    method: "DELETE",
  });
}

async function fetchBasketItems(): Promise<StockListItem[]> {
  const res = await apiFetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/basket`);
  if (!res.ok) return [];
  const items = (await res.json()) as BackendBasketItem[];
  return items.map((item) => {
    const name = resolveStockName(item.ticker, item.ticker_name);
    return {
      code: item.ticker,
      name,
      price: "-",
      change: "-",
      logoText: name.slice(0, 1),
    };
  });
}

async function addFavoriteItem(stock: StockListItem) {
  await apiFetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/favorites`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker: stock.code, ticker_name: stock.name }),
  });
}

async function removeFavoriteItem(code: string) {
  await apiFetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/favorites/${encodeURIComponent(code)}`, {
    method: "DELETE",
  });
}

async function fetchFavoriteItems(): Promise<StockListItem[]> {
  const res = await apiFetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/favorites`);
  if (!res.ok) return [];
  const items = (await res.json()) as BackendBasketItem[];
  return items.map((item) => {
    const name = resolveStockName(item.ticker, item.ticker_name);
    return {
      code: item.ticker,
      name,
      price: "-",
      change: "-",
      logoText: name.slice(0, 1),
    };
  });
}

async function hydrateLivePrice(
  stock: StockListItem,
  setStocks: React.Dispatch<React.SetStateAction<StockListItem[]>>,
) {
  try {
    // /prices/current(KIS inquire-price)는 종목명을 내려주지 않는다(업종명만 있음) --
    // 가격·등락률만 갱신한다. 이름은 fetchBasketItems/fetchFavoriteItems에서 이미
    // resolveStockName으로 처리됨.
    const live = await fetchCurrentPrice(stock.code);
    setStocks((prev) =>
      prev.map((item) =>
        item.code === stock.code
          ? { ...item, price: live.priceFormatted, change: live.changeFormatted }
          : item,
      ),
    );
  } catch (error) {
    // 시세 조회 실패 시 기존 값(바구니 저장값)을 그대로 둔다 -- 한 종목 실패가 나머지를 막지 않음
    console.error(`live price fetch failed for ${stock.code}`, error);
  }
}

function makeFallbackStock(code: string): StockListItem {
  const name = resolveStockName(code, code);
  return {
    code,
    name,
    price: "-",
    change: "-",
    logoText: name.slice(0, 1),
  };
}

function upsertStock(items: StockListItem[], stock: StockListItem) {
  const exists = items.some((item) => item.code === stock.code);
  if (exists) {
    return items.map((item) => (item.code === stock.code ? { ...item, ...stock } : item));
  }
  return [...items, stock];
}

export const StockListProvider = ({ children }: { children: React.ReactNode }) => {
  const [favoriteStocks, setFavoriteStocks] = useState<StockListItem[]>([]);
  const [cartStocks, setCartStocks] = useState<StockListItem[]>([]);

  const [autoTradeUniverse, setAutoTradeUniverse] = useState<Set<string> | null>(null);

  useEffect(() => {
    let cancelled = false;

    // 구글 로그인 콜백처럼 "루트 레이아웃이 먼저 마운트되고 토큰은 비동기로 나중에 쓰이는" 경로에서는
    // 이 useEffect가 토큰이 저장되기 전에 한 번 실행돼버려 관심종목/바구니/AI 유니버스가 전부 401로
    // 빈 채 멈추고, 마운트 1회성이라 토큰이 생긴 뒤에도 다시 안 돌았다(2026-10-28 발견). 마운트 시
    // 1회 + 로그인 완료(AUTH_EVENT_NAME) 시 다시 실행하도록 고쳤다.
    const loadAll = () => {
      // 토큰이 아직 없으면 어차피 401이 확정이라 요청 자체를 보내지 않는다 -- 로그인 완료 시
      // AUTH_EVENT_NAME으로 다시 호출된다.
      if (!getAccessToken()) return;
      void fetchBasketItems().then(async (items) => {
        if (cancelled || items.length === 0) return;
        setCartStocks(items);
        // KIS 시세 조회는 짧은 간격의 연속 호출에 취약하므로 살짝 간격을 두고 순차 조회한다.
        for (const item of items) {
          if (cancelled) return;
          await hydrateLivePrice(item, setCartStocks);
          await new Promise((resolve) => setTimeout(resolve, 300));
        }
      });
      void fetchFavoriteItems().then(async (items) => {
        if (cancelled || items.length === 0) return;
        setFavoriteStocks(items);
        for (const item of items) {
          if (cancelled) return;
          await hydrateLivePrice(item, setFavoriteStocks);
          await new Promise((resolve) => setTimeout(resolve, 300));
        }
      });
      void apiFetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/basket/universe`)
        .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
        .then((data: { tickers: string[] }) => {
          if (!cancelled) setAutoTradeUniverse(new Set(data.tickers));
        })
        .catch((error) => console.error("auto-trade universe load failed", error));
    };

    loadAll();
    window.addEventListener(AUTH_EVENT_NAME, loadAll);
    return () => {
      cancelled = true;
      window.removeEventListener(AUTH_EVENT_NAME, loadAll);
    };
  }, []);

  const favorites = favoriteStocks.map((stock) => stock.code);
  const cart = cartStocks.map((stock) => stock.code);

  const toggleFavorite = (code: string, stock?: StockListItem) => {
    const nextStock = stock ?? makeFallbackStock(code);
    const removing = favoriteStocks.some((item) => item.code === code);

    setFavoriteStocks((prev) => {
      if (removing) {
        return prev.filter((item) => item.code !== code);
      }
      return upsertStock(prev, nextStock);
    });

    const sync = removing ? removeFavoriteItem(code) : addFavoriteItem(nextStock);
    void sync.catch((error) => {
      console.error("favorite sync failed", error);
    });
  };

  const toggleCart = (code: string, stock?: StockListItem) => {
    const nextStock = stock ?? makeFallbackStock(code);
    const removing = cartStocks.some((item) => item.code === code);
    if (!removing && !autoTradeUniverse?.has(code)) {
      window.alert("자동매매 바구니에는 AI 학습 종목(코스피 200)만 담을 수 있습니다.");
      return;
    }
    if (!removing && cartStocks.length >= MAX_CART_STOCKS) {
      window.alert(`자동매매 종목은 최대 ${MAX_CART_STOCKS}개까지 담을 수 있습니다.`);
      return;
    }

    setCartStocks((prev) => {
      if (prev.some((item) => item.code === code)) {
        return prev.filter((item) => item.code !== code);
      }
      return upsertStock(prev, nextStock);
    });

    const sync = removing ? removeBasketItem(code) : addBasketItem(nextStock);
    void sync.catch((error) => {
      console.error("basket sync failed", error);
    });
  };

  const isFavorite = (code: string) => favorites.includes(code);
  const isInCart = (code: string) => cart.includes(code);

  return (
    <StockListContext.Provider value={{ favorites, cart, favoriteStocks, cartStocks, toggleFavorite, toggleCart, isFavorite, isInCart }}>
      {children}
    </StockListContext.Provider>
  );
};

export const useStockList = () => {
  const context = useContext(StockListContext);
  if (!context) {
    throw new Error("useStockList must be used within a StockListProvider");
  }
  return context;
};

"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { apiFetch } from "@/lib/signup/auth";

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
  return items.map((item) => ({
    code: item.ticker,
    name: item.ticker_name,
    price: "-",
    change: "-",
    logoText: item.ticker_name.slice(0, 1),
  }));
}

function makeFallbackStock(code: string): StockListItem {
  return {
    code,
    name: code,
    price: "-",
    change: "-",
    logoText: code.slice(0, 1),
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

  useEffect(() => {
    let cancelled = false;
    void fetchBasketItems().then((items) => {
      if (!cancelled && items.length > 0) setCartStocks(items);
    });
    return () => { cancelled = true; };
  }, []);

  const favorites = favoriteStocks.map((stock) => stock.code);
  const cart = cartStocks.map((stock) => stock.code);

  const toggleFavorite = (code: string, stock?: StockListItem) => {
    setFavoriteStocks((prev) => {
      if (prev.some((item) => item.code === code)) {
        return prev.filter((item) => item.code !== code);
      }
      return upsertStock(prev, stock ?? makeFallbackStock(code));
    });
  };

  const toggleCart = (code: string, stock?: StockListItem) => {
    const nextStock = stock ?? makeFallbackStock(code);
    const removing = cartStocks.some((item) => item.code === code);
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

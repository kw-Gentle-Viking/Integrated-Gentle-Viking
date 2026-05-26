"use client";

import React, { createContext, useContext, useState } from "react";

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

const StockListContext = createContext<StockListContextType | undefined>(undefined);

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
    setCartStocks((prev) => {
      if (prev.some((item) => item.code === code)) {
        return prev.filter((item) => item.code !== code);
      }
      return upsertStock(prev, stock ?? makeFallbackStock(code));
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

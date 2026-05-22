"use client";

import React, { createContext, useContext, useState } from "react";

interface StockListContextType {
  favorites: string[];
  cart: string[];
  toggleFavorite: (code: string) => void;
  toggleCart: (code: string) => void;
  isFavorite: (code: string) => boolean;
  isInCart: (code: string) => boolean;
}

const StockListContext = createContext<StockListContextType | undefined>(undefined);

export const StockListProvider = ({ children }: { children: React.ReactNode }) => {
  const [favorites, setFavorites] = useState<string[]>([]);
  const [cart, setCart] = useState<string[]>([]);

  const toggleFavorite = (code: string) => {
    setFavorites((prev) =>
      prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code],
    );
  };

  const toggleCart = (code: string) => {
    setCart((prev) =>
      prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code],
    );
  };

  const isFavorite = (code: string) => favorites.includes(code);
  const isInCart = (code: string) => cart.includes(code);

  return (
    <StockListContext.Provider value={{ favorites, cart, toggleFavorite, toggleCart, isFavorite, isInCart }}>
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

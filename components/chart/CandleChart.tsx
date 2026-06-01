"use client";

import { useEffect, useRef } from "react";
import { CandlestickSeries, ColorType, createChart, UTCTimestamp } from "lightweight-charts";
import type { CandleType } from "@/lib/chart/types";

interface Props {
  candles: CandleType[];
  height?: number;
}

export default function CandleChart({ candles, height = 280 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || candles.length === 0) return;

    const chart = createChart(el, {
      width: el.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#94a3b8",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#f1f5f9" },
        horzLines: { color: "#f1f5f9" },
      },
      rightPriceScale: { borderColor: "#e2e8f0" },
      timeScale: { borderColor: "#e2e8f0", timeVisible: true },
      handleScroll: {
        mouseWheel: false,      // 스크롤로 좌우 이동 비활성화
        pressedMouseMove: true, // 드래그로 좌우 이동
      },
      handleScale: {
        mouseWheel: true,       // 스크롤로 확대/축소 (캔들 개수 조절)
        pinch: true,
        axisPressedMouseMove: { time: true, price: true },
      },
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#e11d48",       // 상승 = 빨강
      downColor: "#1d4ed8",     // 하락 = 파랑
      borderUpColor: "#e11d48",
      borderDownColor: "#1d4ed8",
      wickUpColor: "#e11d48",
      wickDownColor: "#1d4ed8",
    });

    series.setData(
      candles.map((c) => ({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );

    chart.timeScale().fitContent();

    const onResize = () => chart.applyOptions({ width: el.clientWidth });
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
    };
  }, [candles, height]);

  return <div ref={containerRef} className="w-full" />;
}

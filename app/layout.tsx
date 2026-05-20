import "./globals.css";
import type { Metadata } from "next";
import { SidebarProvider } from "../components/sidebar/SidebarContext";
import { StockListProvider } from "../lib/stock-list/StockListContext";
import Header from "./Layout/Header";
import Footer from "./Layout/Footer";
import LayoutBody from "./Layout/LayoutBody";

export const metadata: Metadata = {
  title: "Gentle Viking",
  description: "AI 포트폴리오 구성 및 자동매매 서비스",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className="bg-white text-[#333D4B] overflow-x-hidden">
        <StockListProvider>
          <SidebarProvider>
            <div className="min-h-screen flex flex-col relative">
              <Header />
              <LayoutBody>{children}</LayoutBody>
              <Footer />
            </div>
          </SidebarProvider>
        </StockListProvider>
      </body>
    </html>
  );
}

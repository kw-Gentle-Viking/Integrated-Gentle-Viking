"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { handleGoogleCallback } from "@/lib/signup/auth";

function GoogleCallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [error, setError] = useState("");

  useEffect(() => {
    handleGoogleCallback(searchParams).then((result) => {
      if (result.ok) {
        router.replace("/dashboard");
      } else {
        setError(result.message);
      }
    });
  }, [router, searchParams]);

  if (error) {
    return (
      <main className="auth-shell flex min-h-screen items-center justify-center px-5">
        <div className="w-full max-w-sm rounded-[26px] border border-white/70 bg-white/90 p-8 text-center shadow-[0_26px_80px_rgba(49,46,129,0.18)] backdrop-blur-2xl">
          <p className="mb-6 text-sm font-bold text-rose-600">{error}</p>
          <button
            type="button"
            onClick={() => router.replace("/login")}
            className="rounded-2xl bg-gradient-to-r from-indigo-600 via-violet-600 to-sky-500 px-6 py-3 text-sm font-black text-white"
          >
            로그인으로 돌아가기
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="auth-shell flex min-h-screen items-center justify-center px-5">
      <div className="text-sm font-bold text-white">
        Google 로그인 처리 중...
      </div>
    </main>
  );
}

export default function GoogleCallbackPage() {
  return (
    <Suspense
      fallback={
        <main className="auth-shell flex min-h-screen items-center justify-center px-5">
          <div className="text-sm font-bold text-white">
            Google 로그인 처리 중...
          </div>
        </main>
      }
    >
      <GoogleCallbackContent />
    </Suspense>
  );
}

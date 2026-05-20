"use client";

import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import {
  IoArrowForward,
  IoChevronBackOutline,
  IoEyeOffOutline,
  IoEyeOutline,
  IoLockClosedOutline,
  IoMailOutline,
} from "react-icons/io5";
import { FcGoogle } from "react-icons/fc";
import { getCurrentUser, googleLogin, loginUser } from "@/lib/signup/auth";

const Login = () => {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);

  const handleGoogleLogin = () => {
    setIsGoogleLoading(true);
    googleLogin();
  };

  useEffect(() => {
    if (getCurrentUser()) {
      router.replace("/dashboard");
    }
  }, [router]);

  const handleLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");

    if (!email.trim() || !password) {
      setError("이메일과 비밀번호를 모두 입력해주세요.");
      return;
    }

    setIsSubmitting(true);
    const result = await loginUser(email, password);
    setIsSubmitting(false);

    if (!result.ok) {
      setError(result.message);
      return;
    }

    router.push("/dashboard");
  };

  return (
    <main className="auth-shell min-h-screen px-5 py-8 text-slate-950">
      <button
        type="button"
        onClick={() => router.back()}
        className="landing-enter landing-enter-1 fixed left-5 top-5 z-30 inline-flex h-11 items-center gap-2 rounded-full border border-white/40 bg-white/20 px-4 text-sm font-black text-white shadow-sm backdrop-blur-xl transition hover:bg-white/30"
      >
        <IoChevronBackOutline className="h-5 w-5" />
        뒤로가기
      </button>

      <section className="mx-auto flex min-h-[calc(100vh-64px)] w-full max-w-[430px] flex-col items-center justify-start pb-10 pt-12 sm:pt-16">
        <div className="landing-enter landing-enter-2 mb-6 flex justify-center">
          <Image
            src="/genvi_white.png"
            alt="Genvi"
            width={108}
            height={32}
            priority
            className="h-auto w-[108px] object-contain"
          />
        </div>
        <form
          onSubmit={handleLogin}
          className="landing-enter landing-enter-4 mx-auto w-full rounded-[26px] border border-white/70 bg-white/90 p-7 shadow-[0_26px_80px_rgba(49,46,129,0.18)] backdrop-blur-2xl sm:p-9"
        >
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-black tracking-normal text-slate-950">
              로그인
            </h1>
            <p className="mt-3 text-xs font-black uppercase tracking-[0.18em] text-indigo-500">
              Welcome back !
            </p>
            <p className="mt-2 text-sm leading-6 text-slate-500">
              데모 계정은 test@gentleviking.com / 1234 입니다.
            </p>
          </div>

          <div>
            <label
              htmlFor="login-email"
              className="mb-2 block text-xs font-black text-slate-600"
            >
              이메일
            </label>
            <div className="relative">
              <IoMailOutline className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
              <input
                id="login-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="test@gentleviking.com"
                autoComplete="email"
                className="h-13 w-full rounded-2xl border border-slate-200 bg-white/80 px-12 py-4 text-sm font-semibold text-slate-900 outline-none placeholder:text-slate-400 focus:border-indigo-400 focus:ring-4 focus:ring-indigo-500/10"
              />
            </div>
          </div>

          <div className="mt-5">
            <label
              htmlFor="login-password"
              className="mb-2 block text-xs font-black text-slate-600"
            >
              비밀번호
            </label>

            <div className="relative">
              <IoLockClosedOutline className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
              <input
                id="login-password"
                type={showPw ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="비밀번호를 입력하세요"
                autoComplete="current-password"
                className="h-13 w-full rounded-2xl border border-slate-200 bg-white/80 px-12 py-4 pr-14 text-sm font-semibold text-slate-900 outline-none placeholder:text-slate-400 focus:border-indigo-400 focus:ring-4 focus:ring-indigo-500/10"
              />

              <button
                type="button"
                onClick={() => setShowPw((v) => !v)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 transition hover:text-slate-700"
                aria-label={showPw ? "비밀번호 숨기기" : "비밀번호 보기"}
              >
                {showPw ? (
                  <IoEyeOffOutline className="h-5 w-5" />
                ) : (
                  <IoEyeOutline className="h-5 w-5" />
                )}
              </button>
            </div>
          </div>

          {error && (
            <p className="mt-4 rounded-2xl bg-rose-50 px-4 py-3 text-xs font-bold text-rose-600">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="mt-7 inline-flex h-13 w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-600 via-violet-600 to-sky-500 px-5 py-4 text-sm font-black text-white shadow-[0_16px_40px_rgba(79,70,229,0.22)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isSubmitting ? "로그인 중..." : "로그인"}
            <IoArrowForward className="h-4 w-4" />
          </button>

          <div className="my-5 flex items-center gap-3 text-xs font-bold text-slate-400">
            <span className="h-px flex-1 bg-slate-200" />
            또는
            <span className="h-px flex-1 bg-slate-200" />
          </div>

          <button
            type="button"
            onClick={handleGoogleLogin}
            disabled={isGoogleLoading || isSubmitting}
            className="inline-flex h-13 w-full items-center justify-center gap-3 rounded-2xl border border-slate-200 bg-white px-5 py-4 text-sm font-black text-slate-700 shadow-sm transition hover:border-indigo-200 hover:bg-indigo-50/40 disabled:cursor-not-allowed disabled:opacity-60"
            aria-label="구글로 로그인"
          >
            <FcGoogle className="h-5 w-5" />
            {isGoogleLoading ? "Google 연결 중..." : "Google로 계속하기"}
          </button>

          <div className="mt-5 text-center text-sm font-semibold text-slate-500">
            아직 계정이 없나요?{" "}
            <Link href="/signup" className="font-black text-indigo-600">
              회원가입
            </Link>
          </div>
        </form>
      </section>
    </main>
  );
};

export default Login;

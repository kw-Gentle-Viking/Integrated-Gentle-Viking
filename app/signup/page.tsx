"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { IoChevronBackOutline } from "react-icons/io5";
import { FcGoogle } from "react-icons/fc";

import { BasicForm, RiskProfile } from "@/lib/signup/types";
import SignupForm from "@/components/signup/SignupForm";
import Survey from "@/components/signup/Survey";
import {
  getCurrentUser,
  googleLogin,
  isEmailRegistered,
  signupUser,
} from "@/lib/signup/auth";

type Step = 1 | 2;

type PwRules = {
  lengthOk: boolean;
  hasLetter: boolean;
  hasNumber: boolean;
  noSpace: boolean;
};

function isEmailValid(email: string) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
}

function isPhoneValid(phone: string) {
  return /^\d{10,11}$/.test(phone.replace(/\D/g, ""));
}

function checkPassword(pw: string): PwRules {
  return {
    lengthOk: pw.length >= 8 && pw.length <= 20,
    hasLetter: /[A-Za-z]/.test(pw),
    hasNumber: /[0-9]/.test(pw),
    noSpace: !/\s/.test(pw),
  };
}

function isPasswordValid(pw: string) {
  const r = checkPassword(pw);
  return r.lengthOk && r.hasLetter && r.hasNumber && r.noSpace;
}

const SignupPage = () => {
  const router = useRouter();

  const [step, setStep] = useState<Step>(1);
  const [submitError, setSubmitError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [emailChecked, setEmailChecked] = useState(false);

  const [basicForm, setBasicForm] = useState<BasicForm>({
    name: "",
    email: "",
    pw1: "",
    pw2: "",
    nickname: "",
    birthdate: "",
    phone: "",
    consentRequired: false,
  });

  const [survey, setSurvey] = useState<RiskProfile>({
    goal: "PRESERVE",
    horizon: "LT3M",
    lossTolerance: "LT5",
    experience: "NONE",
    volatility: "LOW",
  });

  const [touched, setTouched] = useState<Record<keyof BasicForm, boolean>>({
    name: false,
    email: false,
    pw1: false,
    pw2: false,
    nickname: false,
    birthdate: false,
    phone: false,
    consentRequired: false,
  });

  useEffect(() => {
    if (getCurrentUser()) {
      router.replace("/dashboard");
    }
  }, [router]);

  const pwRules = useMemo(() => checkPassword(basicForm.pw1), [basicForm.pw1]);

  const emailIsAlreadyRegistered = useMemo(() => {
    if (!isEmailValid(basicForm.email)) return false;
    return isEmailRegistered(basicForm.email);
  }, [basicForm.email]);

  const basicErrors = useMemo(() => {
    const e: Partial<Record<keyof BasicForm, string>> = {};

    if (!basicForm.name.trim()) e.name = "이름을 입력해주세요.";
    if (!basicForm.nickname.trim()) e.nickname = "닉네임을 입력해주세요.";

    if (!basicForm.email.trim()) e.email = "이메일을 입력해주세요.";
    else if (!isEmailValid(basicForm.email))
      e.email = "이메일 형식이 올바르지 않습니다.";
    else if (emailIsAlreadyRegistered) e.email = "이미 가입된 이메일입니다.";
    else if (!emailChecked) e.email = "이메일 중복체크를 완료해주세요.";

    if (!basicForm.phone.trim()) e.phone = "휴대폰 번호를 입력해주세요.";
    else if (!isPhoneValid(basicForm.phone))
      e.phone = "휴대폰 번호는 숫자만 10~11자로 입력해주세요.";

    if (!basicForm.birthdate) e.birthdate = "생년월일을 선택해주세요.";

    if (!basicForm.pw1) e.pw1 = "비밀번호를 입력해주세요.";
    else if (!isPasswordValid(basicForm.pw1))
      e.pw1 = "비밀번호 조건을 확인해주세요.";
    if (!basicForm.pw2) e.pw2 = "비밀번호를 한 번 더 입력해주세요.";
    else if (basicForm.pw2 !== basicForm.pw1)
      e.pw2 = "비밀번호가 일치하지 않습니다.";

    if (!basicForm.consentRequired)
      e.consentRequired = "필수 동의가 필요합니다.";

    return e;
  }, [basicForm, emailChecked, emailIsAlreadyRegistered]);

  const canGoStep2 = useMemo(
    () => Object.keys(basicErrors).length === 0,
    [basicErrors],
  );

  const surveyScore = useMemo(() => {
    const goal = (
      { PRESERVE: 1, STABLE: 2, GROWTH: 3, AGGRESSIVE: 4 } as const
    )[survey.goal];
    const period = ({ LT3M: 1, M3TO12: 2, Y1TO3: 3, GT3Y: 4 } as const)[
      survey.horizon
    ];
    const tolerance = ({ LT5: 1, LT10: 2, LT20: 3, GT30: 4 } as const)[
      survey.lossTolerance
    ];
    const experience = (
      { NONE: 1, SAVING: 2, STOCK_ETF: 3, DERIV_CRYPTO: 4 } as const
    )[survey.experience];
    const volatility = ({ LOW: 1, MID: 2, HIGH: 3 } as const)[
      survey.volatility
    ];

    const weighted =
      ((goal - 1) / 3) * 100 * 0.2 +
      ((period - 1) / 3) * 100 * 0.1 +
      ((tolerance - 1) / 3) * 100 * 0.3 +
      ((experience - 1) / 3) * 100 * 0.15 +
      ((volatility - 1) / 2) * 100 * 0.25;

    let grade =
      weighted <= 20
        ? 1
        : weighted <= 40
          ? 2
          : weighted <= 60
            ? 3
            : weighted <= 80
              ? 4
              : 5;

    // 손실 감내 최하(LT5)이면 최대 안정추구형(2)으로 제한
    if (tolerance === 1) grade = Math.min(2, grade);
    return grade;
  }, [survey]);

  const surveyLabel = useMemo(() => {
    if (surveyScore <= 1) return "안정형";
    if (surveyScore <= 2) return "안정추구형";
    if (surveyScore <= 3) return "위험중립형";
    if (surveyScore <= 4) return "적극투자형";
    return "공격투자형";
  }, [surveyScore]);

  const setField = <K extends keyof BasicForm>(key: K, value: BasicForm[K]) => {
    if (key === "email") {
      setEmailChecked(false);
      setSubmitError("");
    }

    setBasicForm((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const markTouched = <K extends keyof BasicForm>(key: K) => {
    setTouched((prev) => ({
      ...prev,
      [key]: true,
    }));
  };

  const markAllTouched = () => {
    (Object.keys(basicForm) as (keyof BasicForm)[]).forEach((key) =>
      markTouched(key),
    );
  };

  const onCheckEmail = () => {
    markTouched("email");

    if (!basicForm.email.trim() || !isEmailValid(basicForm.email)) {
      setEmailChecked(false);
      return;
    }

    setEmailChecked(!isEmailRegistered(basicForm.email));
  };

  const onNext = () => {
    if (!canGoStep2) {
      markAllTouched();
      return;
    }
    setSubmitError("");
    setStep(2);
  };

  const onPrev = () => setStep(1);

  const onSubmit = async () => {
    if (isSubmitting) return;
    setIsSubmitting(true);

    const result = await signupUser({
      basic: basicForm,
      riskProfile: survey,
      riskScore: surveyScore,
      riskLabel: surveyLabel,
    });

    setIsSubmitting(false);

    if (!result.ok) {
      setSubmitError(result.message);
      setStep(1);
      markTouched("email");
      return;
    }

    alert(
      `가입이 완료되었습니다.\n\n${result.user.nickname}님의 투자성향: ${result.user.riskLabel} (score: ${result.user.riskScore})`,
    );
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

      <section className="mx-auto flex min-h-[calc(100vh-64px)] w-full max-w-[460px] flex-col items-center justify-start pb-10 pt-12 sm:pt-16">
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

        <section className="landing-enter landing-enter-4 relative w-full rounded-[26px] border border-white/70 bg-white/90 p-7 shadow-[0_26px_80px_rgba(49,46,129,0.18)] backdrop-blur-2xl sm:p-9">
          <span className="absolute right-5 top-5 rounded-full bg-slate-950 px-4 py-2 text-xs font-black text-white">
            {step}/2
          </span>

          <div className="mb-7 text-center">
            <h1 className="text-3xl font-black tracking-normal text-slate-950">
              회원가입
            </h1>
            <p className="mt-3 text-xs font-black uppercase tracking-[0.18em] text-indigo-500">
              {step === 1 ? "Basic information" : "Risk profile"}
            </p>
            <p className="mt-2 text-sm leading-6 text-slate-500">
              {step === 1
                ? "기본 정보를 입력해주세요."
                : "투자 성향에 맞는 화면 구성을 준비합니다."}
            </p>
          </div>

          {submitError && (
            <p className="mb-4 rounded-2xl bg-rose-50 px-4 py-3 text-xs font-bold text-rose-600">
              {submitError}
            </p>
          )}

          {step === 1 && (
            <>
              <button
                type="button"
                onClick={googleLogin}
                className="mb-5 inline-flex h-13 w-full items-center justify-center gap-3 rounded-2xl border border-slate-200 bg-white px-5 py-4 text-sm font-black text-slate-700 shadow-sm transition hover:border-indigo-200 hover:bg-indigo-50/40"
                aria-label="구글로 회원가입"
              >
                <FcGoogle className="h-5 w-5" />
                Google로 계속하기
              </button>

              <div className="mb-5 flex items-center gap-3 text-xs font-bold text-slate-400">
                <span className="h-px flex-1 bg-slate-200" />
                또는
                <span className="h-px flex-1 bg-slate-200" />
              </div>
            </>
          )}

          {step === 1 ? (
            <SignupForm
              value={basicForm}
              touched={touched}
              errors={basicErrors}
              pwRules={pwRules}
              onChange={setField}
              onBlur={markTouched}
              onBack={() => router.back()}
              onNext={onNext}
              nextDisabled={!canGoStep2}
              emailChecked={emailChecked}
              emailIsAlreadyRegistered={emailIsAlreadyRegistered}
              onCheckEmail={onCheckEmail}
            />
          ) : (
            <Survey
              value={survey}
              onChange={setSurvey}
              score={surveyScore}
              label={surveyLabel}
              onPrev={onPrev}
              onSubmit={onSubmit}
              isSubmitting={isSubmitting}
            />
          )}

          <div className="mt-7 flex justify-center gap-2">
            {[1, 2].map((n) => (
              <span
                key={n}
                className={[
                  "h-2.5 rounded-full border transition-all",
                  step === n
                    ? "w-8 border-indigo-600 bg-indigo-600 opacity-100"
                    : "border-slate-300 bg-transparent opacity-40",
                ].join(" ")}
              />
            ))}
          </div>
        </section>
      </section>
    </main>
  );
};

export default SignupPage;

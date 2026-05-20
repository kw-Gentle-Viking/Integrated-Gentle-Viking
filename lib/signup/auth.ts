import { BasicForm, RiskProfile } from "@/lib/signup/types";

export const AUTH_EVENT_NAME = "gv-auth-change";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const USERS_KEY = "gv-users";
const SESSION_KEY = "gv-session";
const LEGACY_AUTH_KEY = "gv-auth";
const ACCESS_TOKEN_KEY = "gv-access-token";
const REFRESH_TOKEN_KEY = "gv-refresh-token";

export type StoredUser = {
  id: string;
  name: string;
  email: string;
  password?: string;
  nickname: string;
  birthdate: string;
  phone: string;
  consentRequired: boolean;
  riskProfile: RiskProfile;
  riskScore: number;
  riskLabel: string;
  createdAt: string;
};

export type AuthUser = Omit<StoredUser, "password">;

type LoginResult =
  | { ok: true; user: AuthUser }
  | { ok: false; message: string };

type SignupResult =
  | { ok: true; user: AuthUser }
  | { ok: false; message: string };

type SignupApiResponse = {
  id: number;
  email: string;
  name: string;
  nickname: string;
  phone: string;
  birth_date: string;
  risk_score: number;
  created_at: string;
};

const DEMO_USER: StoredUser = {
  id: "demo-user",
  name: "테스트 사용자",
  email: "test@gentleviking.com",
  password: "1234",
  nickname: "젠틀바이킹",
  birthdate: "1990-01-01",
  phone: "01012345678",
  consentRequired: true,
  riskProfile: {
    goal: "STABLE",
    horizon: "Y1TO3",
    lossTolerance: "LT10",
    experience: "STOCK_ETF",
    volatility: "MID",
  },
  riskScore: 3,
  riskLabel: "위험중립형",
  createdAt: "2026-01-01T00:00:00.000Z",
};

function canUseStorage() {
  return typeof window !== "undefined" && Boolean(window.localStorage);
}

function normalizeEmail(email: string) {
  return email.trim().toLowerCase();
}

function toAuthUser(user: StoredUser): AuthUser {
  const safeUser = { ...user };
  delete (safeUser as Partial<StoredUser>).password;
  return safeUser as AuthUser;
}

function notifyAuthChange() {
  if (!canUseStorage()) return;
  window.dispatchEvent(new Event(AUTH_EVENT_NAME));
}

function readUsers(): StoredUser[] {
  if (!canUseStorage()) return [DEMO_USER];

  const rawUsers = window.localStorage.getItem(USERS_KEY);
  if (!rawUsers) {
    window.localStorage.setItem(USERS_KEY, JSON.stringify([DEMO_USER]));
    return [DEMO_USER];
  }

  try {
    const users = JSON.parse(rawUsers) as StoredUser[];
    if (!Array.isArray(users)) return [DEMO_USER];

    const hasDemoUser = users.some(
      (user) => normalizeEmail(user.email) === DEMO_USER.email,
    );

    if (hasDemoUser) return users;

    const nextUsers = [DEMO_USER, ...users];
    window.localStorage.setItem(USERS_KEY, JSON.stringify(nextUsers));
    return nextUsers;
  } catch {
    window.localStorage.setItem(USERS_KEY, JSON.stringify([DEMO_USER]));
    return [DEMO_USER];
  }
}

function writeUsers(users: StoredUser[]) {
  if (!canUseStorage()) return;
  const sanitized = users.map(({ password: _pw, ...rest }) => {
    void _pw;
    return rest;
  });
  window.localStorage.setItem(USERS_KEY, JSON.stringify(sanitized));
}

function writeSession(email: string) {
  if (!canUseStorage()) return;
  window.localStorage.setItem(SESSION_KEY, normalizeEmail(email));
  window.localStorage.setItem(LEGACY_AUTH_KEY, "true");
  notifyAuthChange();
}

export function isEmailRegistered(email: string) {
  const targetEmail = normalizeEmail(email);
  if (!targetEmail) return false;

  return readUsers().some((user) => normalizeEmail(user.email) === targetEmail);
}

const RISK_INT_MAP = {
  goal:          { PRESERVE: 1, STABLE: 2, GROWTH: 3, AGGRESSIVE: 4 },
  horizon:       { LT3M: 1, M3TO12: 2, Y1TO3: 3, GT3Y: 4 },
  lossTolerance: { LT5: 1, LT10: 2, LT20: 3, GT30: 4 },
  experience:    { NONE: 1, SAVING: 2, STOCK_ETF: 3, DERIV_CRYPTO: 4 },
  volatility:    { LOW: 1, MID: 2, HIGH: 3 },
} as const;

function riskScoreToLabel(score: number): string {
  if (score <= 1) return "안정형";
  if (score <= 2) return "안정추구형";
  if (score <= 3) return "위험중립형";
  if (score <= 4) return "적극투자형";
  return "공격투자형";
}

export async function signupUser({
  basic,
  riskProfile,
}: {
  basic: BasicForm;
  riskProfile: RiskProfile;
  riskScore: number;
  riskLabel: string;
}): Promise<SignupResult> {
  const email = normalizeEmail(basic.email);

  try {
    const res = await fetch(`${API_BASE}/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name:                  basic.name.trim(),
        nickname:              basic.nickname.trim(),
        email,
        phone:                 basic.phone.replace(/\D/g, ""),
        password:              basic.pw1,
        birth_date:            basic.birthdate,
        investment_goal:       RISK_INT_MAP.goal[riskProfile.goal],
        investment_period:     RISK_INT_MAP.horizon[riskProfile.horizon],
        risk_tolerance:        RISK_INT_MAP.lossTolerance[riskProfile.lossTolerance],
        investment_experience: RISK_INT_MAP.experience[riskProfile.experience],
        volatility_preference: RISK_INT_MAP.volatility[riskProfile.volatility],
      }),
    });

    if (res.status === 409) {
      return { ok: false, message: "이미 가입된 이메일입니다. 로그인하거나 다른 이메일을 사용해주세요." };
    }
    if (res.status === 422) {
      return { ok: false, message: "입력값을 확인해주세요." };
    }
    if (!res.ok) {
      return { ok: false, message: "회원가입 중 오류가 발생했습니다." };
    }

    const data = await res.json() as SignupApiResponse;
    const riskLabel = riskScoreToLabel(data.risk_score);

    const user: StoredUser = {
      id:              String(data.id),
      name:            data.name,
      email:           normalizeEmail(data.email),
      nickname:        data.nickname,
      birthdate:       data.birth_date,
      phone:           data.phone,
      consentRequired: basic.consentRequired,
      riskProfile,
      riskScore:       data.risk_score,
      riskLabel,
      createdAt:       data.created_at,
    };

    writeUsers([...readUsers(), user]);
    writeSession(user.email);

    return { ok: true, user: toAuthUser(user) };
  } catch {
    return { ok: false, message: "서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요." };
  }
}

export async function loginUser(
  email: string,
  password: string,
): Promise<LoginResult> {
  // 데모 계정은 API 없이 로컬에서 처리
  if (normalizeEmail(email) === normalizeEmail(DEMO_USER.email)) {
    if (password !== DEMO_USER.password) {
      return {
        ok: false,
        message: "이메일 또는 비밀번호가 올바르지 않습니다.",
      };
    }
    writeSession(DEMO_USER.email);
    return { ok: true, user: toAuthUser(DEMO_USER) };
  }

  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (res.status === 401) {
      return {
        ok: false,
        message: "이메일 또는 비밀번호가 올바르지 않습니다.",
      };
    }
    if (res.status === 422) {
      return { ok: false, message: "입력값을 확인해주세요." };
    }
    if (!res.ok) {
      return { ok: false, message: "로그인 중 오류가 발생했습니다." };
    }

    const tokens = (await res.json()) as {
      access_token: string;
      refresh_token: string;
      token_type: string;
    };

    if (canUseStorage()) {
      window.localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token);
      window.localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token);
    }

    const normalizedEmail = normalizeEmail(email);
    writeSession(normalizedEmail);

    const localUser = readUsers().find(
      (u) => normalizeEmail(u.email) === normalizedEmail,
    );
    if (localUser) {
      return { ok: true, user: toAuthUser(localUser) };
    }

    // 백엔드에만 존재하는 유저 — JWT에서 기본 정보 추출
    let userId = normalizedEmail;
    try {
      const payload = JSON.parse(atob(tokens.access_token.split(".")[1]));
      userId = payload.sub ?? userId;
    } catch {
      /* JWT 디코딩 실패 시 이메일 사용 */
    }

    const fallbackUser: AuthUser = {
      id: userId,
      name: normalizedEmail.split("@")[0],
      email: normalizedEmail,
      nickname: normalizedEmail.split("@")[0],
      birthdate: "",
      phone: "",
      consentRequired: true,
      riskProfile: {
        goal: "PRESERVE",
        horizon: "LT3M",
        lossTolerance: "LT5",
        experience: "NONE",
        volatility: "LOW",
      },
      riskScore: 5,
      riskLabel: "안정형",
      createdAt: new Date().toISOString(),
    };

    const existingUsers = readUsers();
    const alreadyStored = existingUsers.some(
      (u) => normalizeEmail(u.email) === normalizedEmail,
    );
    if (!alreadyStored) {
      writeUsers([...existingUsers, fallbackUser as StoredUser]);
    }

    return { ok: true, user: fallbackUser };
  } catch {
    return {
      ok: false,
      message: "서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.",
    };
  }
}

export async function logoutUser() {
  const refreshToken = canUseStorage()
    ? window.localStorage.getItem(REFRESH_TOKEN_KEY)
    : null;

  if (refreshToken) {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
    } catch {
      // 서버 오류여도 클라이언트 토큰은 항상 삭제
    }
  }

  if (!canUseStorage()) return;
  window.localStorage.removeItem(SESSION_KEY);
  window.localStorage.removeItem(LEGACY_AUTH_KEY);
  window.localStorage.removeItem(ACCESS_TOKEN_KEY);
  window.localStorage.removeItem(REFRESH_TOKEN_KEY);
  notifyAuthChange();
}

export function getAccessToken(): string | null {
  if (!canUseStorage()) return null;
  return window.localStorage.getItem(ACCESS_TOKEN_KEY);
}

// access token 만료 시 refresh token으로 새 토큰 발급
export async function refreshTokens(): Promise<boolean> {
  const refreshToken = canUseStorage()
    ? window.localStorage.getItem(REFRESH_TOKEN_KEY)
    : null;

  if (!refreshToken) return false;

  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!res.ok) {
      // refresh token도 만료 또는 무효 → 강제 로그아웃
      await logoutUser();
      return false;
    }

    const tokens = (await res.json()) as {
      access_token: string;
      refresh_token: string;
      token_type: string;
    };

    if (canUseStorage()) {
      window.localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token);
      window.localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token);
    }

    return true;
  } catch {
    return false;
  }
}

// 인증이 필요한 API 호출 래퍼 — 401 시 토큰 갱신 후 1회 재시도
export async function apiFetch(
  input: RequestInfo,
  init?: RequestInit,
): Promise<Response> {
  const accessToken = getAccessToken();

  const headers = new Headers(init?.headers);
  if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);

  const res = await fetch(input, { ...init, headers });

  if (res.status !== 401) return res;

  // 데모 계정은 토큰 갱신 없이 그대로 반환
  const sessionEmail = canUseStorage()
    ? window.localStorage.getItem(SESSION_KEY)
    : null;
  if (sessionEmail === normalizeEmail(DEMO_USER.email)) return res;

  // access token 만료 → refresh 시도
  const refreshed = await refreshTokens();
  if (!refreshed) return res;

  // 새 토큰으로 재시도
  const newToken = getAccessToken();
  const retryHeaders = new Headers(init?.headers);
  if (newToken) retryHeaders.set("Authorization", `Bearer ${newToken}`);

  return fetch(input, { ...init, headers: retryHeaders });
}

export function getCurrentUser(): AuthUser | null {
  if (!canUseStorage()) return null;

  const sessionEmail = window.localStorage.getItem(SESSION_KEY);
  if (!sessionEmail) return null;

  const matchedUser = readUsers().find(
    (user) => normalizeEmail(user.email) === normalizeEmail(sessionEmail),
  );

  return matchedUser ? toAuthUser(matchedUser) : null;
}

export function googleLogin(): void {
  window.location.href = `${API_BASE}/auth/google/login`;
}

export async function handleGoogleCallback(
  params: URLSearchParams,
): Promise<LoginResult> {
  const accessToken = params.get('access_token');
  const refreshToken = params.get('refresh_token');

  if (!accessToken) {
    return { ok: false, message: 'Google 로그인에 실패했습니다.' };
  }

  if (canUseStorage()) {
    window.localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
    if (refreshToken) {
      window.localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
    }
  }

  let email = '';
  let userId = '';
  try {
    const payload = JSON.parse(atob(accessToken.split('.')[1]));
    userId = payload.sub ?? '';
    email = payload.email ?? payload.sub ?? '';
  } catch { /* JWT 디코딩 실패 시 빈 값 유지 */ }

  if (!email) {
    return { ok: false, message: 'Google 로그인에 실패했습니다.' };
  }

  const normalizedEmail = normalizeEmail(email);
  writeSession(normalizedEmail);

  const localUser = readUsers().find(
    (u) => normalizeEmail(u.email) === normalizedEmail,
  );
  if (localUser) return { ok: true, user: toAuthUser(localUser) };

  const fallbackUser: AuthUser = {
    id:              userId || normalizedEmail,
    name:            normalizedEmail.split('@')[0],
    email:           normalizedEmail,
    nickname:        normalizedEmail.split('@')[0],
    birthdate:       '',
    phone:           '',
    consentRequired: false,
    riskProfile:     { goal: 'PRESERVE', horizon: 'LT3M', lossTolerance: 'LT5', experience: 'NONE', volatility: 'LOW' },
    riskScore:       5,
    riskLabel:       '안정형',
    createdAt:       new Date().toISOString(),
  };

  writeUsers([...readUsers(), fallbackUser as StoredUser]);
  return { ok: true, user: fallbackUser };
}

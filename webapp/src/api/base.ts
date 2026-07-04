/**
 * Backend base URL resolution.
 *
 * In the browser the app is served behind nginx, which proxies /api and /ws
 * to the engine, so relative URLs work. In the native iOS app (Capacitor)
 * the origin is capacitor://localhost, so relative URLs would hit the local
 * webview server instead of the backend — native builds must use the
 * absolute production host.
 *
 * Uses globalThis.location (not window) so this module is safe to import
 * from Web Worker code as well.
 */

const NATIVE_BACKEND_HOST = 'knightball.org';

export const isNativeApp = globalThis.location?.protocol === 'capacitor:';

/** Absolute backend origin on native, '' in the browser (relative URLs). */
export const BACKEND_ORIGIN = isNativeApp ? `https://${NATIVE_BACKEND_HOST}` : '';

export const API_BASE = `${BACKEND_ORIGIN}/api`;

/**
 * Anonymous identity for the native app.
 *
 * WKWebView's tracking prevention drops cross-site cookies (even
 * SameSite=None), so the server's cookie-based anon session can't work from
 * capacitor://localhost. Instead the native app generates a stable anon id,
 * sends it as an X-Anon-Id header on API requests and an ?anon_id= query
 * param on the WS handshake. Same trust level as the cookie (both are
 * client-supplied). Browser builds keep using cookies.
 */
const ANON_ID_KEY = 'knightball_native_anon_id';

export function nativeAnonId(): string | null {
  if (!isNativeApp) return null;
  let id = localStorage.getItem(ANON_ID_KEY);
  if (!id) {
    id = crypto.randomUUID().replace(/-/g, '');
    localStorage.setItem(ANON_ID_KEY, id);
  }
  return id;
}

/**
 * Session token for the native app (accounts). Web uses the HttpOnly cookie;
 * native can't persist cross-site cookies, so the server hands native clients
 * the JWT in auth responses and accepts it via Authorization: Bearer.
 */
const AUTH_TOKEN_KEY = 'knightball_native_auth_token';

export function getNativeAuthToken(): string | null {
  if (!isNativeApp) return null;
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

export function setNativeAuthToken(token: string | null): void {
  if (!isNativeApp) return;
  if (token) localStorage.setItem(AUTH_TOKEN_KEY, token);
  else localStorage.removeItem(AUTH_TOKEN_KEY);
}

/**
 * PKCE for the native OAuth deep-link return (login-CSRF hardening).
 *
 * A crafted knightball://auth?ticket=<attacker's ticket> link must not sign
 * this app into the attacker's account. The app generates a verifier before
 * opening the system browser and sends SHA-256(verifier) with the flow; the
 * ticket exchange then requires the original verifier, which only this app
 * instance holds — a planted ticket can never match.
 */
const PKCE_VERIFIER_KEY = 'knightball_native_pkce_verifier';

function base64url(bytes: Uint8Array): string {
  let s = '';
  bytes.forEach((b) => { s += String.fromCharCode(b); });
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** Generate + persist a fresh PKCE verifier; returns its SHA-256 challenge. */
export async function beginPkceFlow(): Promise<string> {
  const raw = new Uint8Array(32);
  crypto.getRandomValues(raw);
  const verifier = base64url(raw);
  localStorage.setItem(PKCE_VERIFIER_KEY, verifier);
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier));
  return base64url(new Uint8Array(digest));
}

/** Take (and clear) the pending PKCE verifier, or null if no flow is pending. */
export function takePkceVerifier(): string | null {
  const v = localStorage.getItem(PKCE_VERIFIER_KEY);
  localStorage.removeItem(PKCE_VERIFIER_KEY);
  return v;
}

/**
 * Patch global fetch so every request to the backend carries the native
 * identity: X-Native-Client (server returns tokens in auth responses),
 * Authorization: Bearer when logged in, and X-Anon-Id otherwise.
 * No-op in the browser. Call once at app startup, before any API use.
 */
export function installNativeIdentity(): void {
  if (!isNativeApp) return;
  const anonId = nativeAnonId()!;
  const origFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (url.startsWith(BACKEND_ORIGIN)) {
      const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
      headers.set('X-Native-Client', '1');
      const token = getNativeAuthToken();
      if (token) headers.set('Authorization', `Bearer ${token}`);
      headers.set('X-Anon-Id', anonId);
      init = { ...init, headers };
    }
    return origFetch(input, init);
  };
}

/**
 * WebSocket URL for a game (local or online).
 *
 * Native + logged in: the handshake can't carry cookies or headers, so a
 * one-time short-lived ticket is fetched from /auth/ws-ticket (hence async).
 * Native + anonymous: ?anon_id=. Browser: cookies, relative URL.
 */
export async function gameWebSocketUrl(gameId: string): Promise<string> {
  if (isNativeApp) {
    const wsBase = `wss://${NATIVE_BACKEND_HOST}/ws/games/${gameId}/ws`;
    if (getNativeAuthToken()) {
      try {
        const resp = await fetch(`${API_BASE}/auth/ws-ticket`, { method: 'POST', credentials: 'include' });
        if (resp.ok) {
          const { ticket } = await resp.json();
          return `${wsBase}?ticket=${encodeURIComponent(ticket)}`;
        }
      } catch { /* fall through to anonymous */ }
    }
    return `${wsBase}?anon_id=${nativeAnonId()}`;
  }
  const wsProtocol = globalThis.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsProtocol}//${globalThis.location.host}/ws/games/${gameId}/ws`;
}

/**
 * Origin to use for user-facing share links (e.g. online-game join links).
 * Links shared from the native app must point at the public website, not
 * capacitor://localhost.
 */
export const SHARE_ORIGIN = isNativeApp
  ? `https://${NATIVE_BACKEND_HOST}`
  : globalThis.location?.origin ?? '';

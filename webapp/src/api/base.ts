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
 * Patch global fetch so every request to the backend carries X-Anon-Id.
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
      headers.set('X-Anon-Id', anonId);
      init = { ...init, headers };
    }
    return origFetch(input, init);
  };
}

/** WebSocket URL for a game (local or online). */
export function gameWebSocketUrl(gameId: string): string {
  if (isNativeApp) {
    const anonParam = `?anon_id=${nativeAnonId()}`;
    return `wss://${NATIVE_BACKEND_HOST}/ws/games/${gameId}/ws${anonParam}`;
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

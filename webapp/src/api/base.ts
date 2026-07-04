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

/** WebSocket URL for a game (local or online). */
export function gameWebSocketUrl(gameId: string): string {
  if (isNativeApp) {
    return `wss://${NATIVE_BACKEND_HOST}/ws/games/${gameId}/ws`;
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

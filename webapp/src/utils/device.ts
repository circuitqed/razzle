import { isNativeApp } from '../api/base';

/**
 * Desktop/laptop browser (not the native app, not a phone or tablet). These
 * get the deep-search ladder levels, which take tens of seconds per move on
 * a phone. Same mobile test as the AI worker's iOS detection.
 */
export function isDesktopWeb(): boolean {
  if (isNativeApp) return false;
  const nav = globalThis.navigator;
  if (!nav) return false;
  if (/iPhone|iPad|iPod|Android|Mobile/i.test(nav.userAgent)) return false;
  if (nav.platform === 'MacIntel' && nav.maxTouchPoints > 1) return false; // iPadOS
  return true;
}

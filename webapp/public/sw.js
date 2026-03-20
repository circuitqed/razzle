// KnightBall Service Worker — enables offline play and PWA install.
//
// Strategy:
// - HTML navigation: NETWORK-FIRST (always get fresh HTML from server,
//   fall back to cache only when offline). Prevents stale JS references.
// - Fingerprinted assets (/assets/*, .wasm): CACHE-FIRST (immutable filenames)
// - API/WebSocket/ONNX: pass through (no caching)

const CACHE_VERSION = 'knightball-v3';
const STATIC_CACHE = CACHE_VERSION + '-static';

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  // Clean up old caches from previous versions
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((name) => name.startsWith('knightball-') && name !== STATIC_CACHE)
          .map((name) => caches.delete(name))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  if (event.request.method !== 'GET') return;
  if (url.origin !== self.location.origin) return;

  // Pass through: API, WebSocket, ONNX models
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/') || url.pathname.endsWith('.onnx')) {
    return;
  }

  // Network-first for HTML navigation
  if (event.request.mode === 'navigate' || url.pathname === '/') {
    event.respondWith(networkFirst(event.request));
    return;
  }

  // Cache-first for fingerprinted static assets
  if (url.pathname.startsWith('/assets/') || url.pathname.endsWith('.wasm') || url.pathname.endsWith('.woff2')) {
    event.respondWith(cacheFirst(event.request));
    return;
  }

  // Cache-first for other static files (SVG, images, manifest)
  event.respondWith(cacheFirst(event.request));
});

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    return cached || new Response('Offline', { status: 503 });
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response('Offline', { status: 503 });
  }
}

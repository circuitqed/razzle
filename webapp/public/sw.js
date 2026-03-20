// Self-destructing service worker: clears all caches, unregisters itself,
// and forces all open tabs to reload with fresh content from the server.
// This fixes the stale cache issue from the previous PWA implementation.

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    // Clear all caches
    caches.keys()
      .then(names => Promise.all(names.map(name => caches.delete(name))))
      .then(() => self.clients.claim())
      .then(() => self.clients.matchAll({ type: 'window' }))
      .then(clients => {
        // Force all tabs to reload
        for (const client of clients) {
          client.navigate(client.url);
        }
      })
      .then(() => {
        // Unregister this service worker
        self.registration.unregister();
      })
  );
});

// Pass through all fetch requests to the network (no caching)
self.addEventListener('fetch', () => {
  // Do nothing — let the browser handle it normally
});

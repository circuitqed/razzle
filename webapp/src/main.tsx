import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Register service worker in production for PWA support + offline play.
// Uses network-first for HTML to prevent stale cache issues on deploy.
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').then((registration) => {
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        if (!newWorker) return;
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            showUpdateBanner();
          }
        });
      });
    });
  });
}

function showUpdateBanner() {
  const banner = document.createElement('div');
  banner.style.cssText =
    'position:fixed;bottom:16px;left:50%;transform:translateX(-50%);' +
    'background:#1e40af;color:white;padding:10px 20px;border-radius:8px;' +
    'font-family:system-ui;font-size:14px;z-index:9999;display:flex;' +
    'align-items:center;gap:12px;box-shadow:0 4px 12px rgba(0,0,0,0.3)';
  banner.innerHTML =
    '<span>New version available</span>' +
    '<button style="background:#3b82f6;border:none;color:white;padding:4px 12px;' +
    'border-radius:4px;cursor:pointer;font-size:14px" id="sw-update-btn">Update</button>' +
    '<button style="background:none;border:none;color:#93c5fd;cursor:pointer;' +
    'font-size:12px" id="sw-dismiss-btn">Later</button>';
  document.body.appendChild(banner);
  document.getElementById('sw-update-btn')?.addEventListener('click', () => location.reload());
  document.getElementById('sw-dismiss-btn')?.addEventListener('click', () => banner.remove());
}

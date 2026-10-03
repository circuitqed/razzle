import { useCallback, useRef } from 'react';

/**
 * Accessible modal behavior, attached via the returned callback ref on the
 * dialog's root element:
 *
 * - focus moves into the dialog on open (an autoFocus field keeps focus;
 *   otherwise the dialog itself takes it, so the iOS keyboard doesn't pop)
 * - Tab / Shift+Tab cycle within the dialog instead of escaping behind it
 * - Escape calls onClose (omit onClose for dialogs that shouldn't close)
 * - focus returns to whatever was focused before, on close
 *
 * Only the topmost open dialog reacts, so nested dialogs behave. The ref is
 * stable, so it attaches when the element mounts and cleans up on unmount —
 * safe for components that return null while closed.
 */

const FOCUSABLE = [
  'a[href]', 'button:not([disabled])', 'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])', 'textarea:not([disabled])', '[tabindex]:not([tabindex="-1"])',
].join(',');

const openDialogs: HTMLElement[] = [];

export function useDialogA11y(onClose?: () => void): (el: HTMLElement | null) => void {
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const cleanupRef = useRef<(() => void) | null>(null);

  return useCallback((el: HTMLElement | null) => {
    cleanupRef.current?.();
    cleanupRef.current = null;
    if (!el) return;

    const previouslyFocused = document.activeElement as HTMLElement | null;
    openDialogs.push(el);
    if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '-1');
    if (!el.contains(document.activeElement)) el.focus({ preventScroll: true });

    const onKeyDown = (e: KeyboardEvent) => {
      if (openDialogs[openDialogs.length - 1] !== el) return;
      if (e.key === 'Escape' && onCloseRef.current) {
        e.stopPropagation();
        onCloseRef.current();
        return;
      }
      if (e.key !== 'Tab') return;
      const items = Array.from(el.querySelectorAll<HTMLElement>(FOCUSABLE))
        .filter((n) => n.getClientRects().length > 0);
      if (items.length === 0) {
        e.preventDefault();
        el.focus();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || active === el || !el.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !el.contains(active))) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKeyDown, true);

    cleanupRef.current = () => {
      document.removeEventListener('keydown', onKeyDown, true);
      const i = openDialogs.lastIndexOf(el);
      if (i !== -1) openDialogs.splice(i, 1);
      if (previouslyFocused && document.contains(previouslyFocused)) {
        previouslyFocused.focus({ preventScroll: true });
      }
    };
  }, []);
}

/**
 * Haptic feedback for the native app (no-op in the browser).
 * Independent of the sound setting — haptics are silent.
 */

import { Haptics, ImpactStyle, NotificationType } from '@capacitor/haptics';
import { isNativeApp } from '../api/base';

/** Light tap for a move landing on the board. */
export function hapticMove(): void {
  if (!isNativeApp) return;
  Haptics.impact({ style: ImpactStyle.Light }).catch(() => {});
}

/** Win/loss buzz at the end of a game. */
export function hapticGameOver(won: boolean): void {
  if (!isNativeApp) return;
  Haptics.notification({ type: won ? NotificationType.Success : NotificationType.Error }).catch(() => {});
}

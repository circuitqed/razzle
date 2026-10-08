import { useEffect, useRef, useState } from 'react';
import { formatSeconds } from '../utils/searchSpeed';
import { isNativeApp } from '../api/base';
import { NATIVE_SEARCH_BUDGET_MS } from '../hooks/useGame';

interface Props {
  progress: { simsDone: number; totalSims: number } | null;
}

/** Below this, a search is quick enough that a countdown is just noise. */
const SHOW_REMAINING_AFTER_MS = 3000;

/**
 * "thinking 40% · ~12 s left". The time left is extrapolated from this
 * search's own progress, so it adapts to the device and the position;
 * it appears only once a search has run for a few seconds.
 */
export default function ThinkingIndicator({ progress }: Props) {
  const startRef = useRef<number>(performance.now());
  const lastDoneRef = useRef<number>(0);
  const [now, setNow] = useState(() => performance.now());

  // A new search (progress restarts) resets the clock.
  const done = progress?.simsDone ?? 0;
  if (done < lastDoneRef.current) startRef.current = performance.now();
  lastDoneRef.current = done;

  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const total = progress?.totalSims ?? 0;
  const pct = total > 0 ? Math.min(99, Math.round((100 * done) / total)) : null;
  const elapsed = now - startRef.current;
  let remaining: string | null = null;
  if (total > 0 && done > 0 && elapsed >= SHOW_REMAINING_AFTER_MS) {
    let secsLeft = ((elapsed / done) * (total - done)) / 1000;
    // The native app stops searching at its time budget.
    if (isNativeApp) secsLeft = Math.min(secsLeft, (NATIVE_SEARCH_BUDGET_MS - elapsed) / 1000);
    if (secsLeft >= 1) remaining = `~${formatSeconds(secsLeft)} left`;
  }

  return (
    <span className="text-blue-400 text-sm animate-pulse">
      thinking{pct != null ? ` ${pct}%` : '...'}
      {remaining && <span className="text-gray-400"> · {remaining}</span>}
    </span>
  );
}

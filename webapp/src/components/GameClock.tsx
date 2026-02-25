/**
 * GameClock components - chess-clock style display for timed PvP games.
 * PlayerClock: single player's clock, placed above/below the board.
 * CorrespondenceClock: deadline display for correspondence games.
 * GameClock: both clocks side by side (legacy, kept for compatibility).
 */

import { useState, useEffect, useRef } from 'react';

function formatTime(seconds: number): string {
  if (seconds <= 0) return '0:00';
  if (seconds < 30) {
    // Show tenths under 30 seconds
    const secs = Math.floor(seconds);
    const tenths = Math.floor((seconds - secs) * 10);
    return `${secs}.${tenths}`;
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

interface PlayerClockProps {
  time: number;
  label: string;
  isActive: boolean;
  gameOver: boolean;
}

export function PlayerClock({ time, label, isActive, gameOver }: PlayerClockProps) {
  // Tick locally when active: decrement every 100ms for smooth display.
  // Use a ref + timestamp approach so the interval doesn't drift.
  const [displayTime, setDisplayTime] = useState(time);
  const startRef = useRef<{ serverTime: number; localMs: number } | null>(null);

  // Reset to server value whenever time prop changes
  useEffect(() => {
    setDisplayTime(time);
    startRef.current = { serverTime: time, localMs: performance.now() };
  }, [time]);

  // Tick down when active
  useEffect(() => {
    if (!isActive || gameOver || time <= 0) return;
    const interval = setInterval(() => {
      if (!startRef.current) return;
      const elapsed = (performance.now() - startRef.current.localMs) / 1000;
      setDisplayTime(Math.max(0, startRef.current.serverTime - elapsed));
    }, 100);
    return () => clearInterval(interval);
  }, [isActive, gameOver, time]);

  const isLow = displayTime < 30;
  const isExpired = displayTime <= 0;

  return (
    <div
      className={`flex items-center justify-between px-3 py-1.5 rounded font-mono text-sm w-full ${
        isActive && !gameOver
          ? 'bg-gray-700 ring-1 ring-green-500'
          : 'bg-gray-800'
      }`}
    >
      <span className="text-xs text-gray-400">{label}</span>
      <span
        className={`font-bold text-lg ${
          isExpired
            ? 'text-red-500'
            : isLow
            ? 'text-red-400'
            : 'text-white'
        } ${isActive && isLow && !gameOver ? 'animate-pulse' : ''}`}
      >
        {formatTime(displayTime)}
      </span>
    </div>
  );
}

function formatDeadline(deadline: string): string {
  const now = Date.now();
  const target = new Date(deadline).getTime();
  const diffMs = target - now;

  if (diffMs <= 0) return 'Expired';

  const totalMinutes = Math.floor(diffMs / 60000);
  const totalHours = Math.floor(totalMinutes / 60);
  const days = Math.floor(totalHours / 24);
  const hours = totalHours % 24;
  const minutes = totalMinutes % 60;

  if (days > 0) return `${days}d ${hours}h left`;
  if (hours > 0) return `${hours}h ${minutes}m left`;
  return `${minutes}m left`;
}

interface CorrespondenceClockProps {
  deadline: string;
  label: string;
  isActive: boolean;
  gameOver: boolean;
}

export function CorrespondenceClock({ deadline, label, isActive, gameOver }: CorrespondenceClockProps) {
  const [display, setDisplay] = useState(() => formatDeadline(deadline));

  useEffect(() => {
    setDisplay(formatDeadline(deadline));
    const interval = setInterval(() => {
      setDisplay(formatDeadline(deadline));
    }, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [deadline]);

  const now = Date.now();
  const target = new Date(deadline).getTime();
  const diffMs = target - now;
  const isUrgent = diffMs > 0 && diffMs < 3600000; // Under 1 hour
  const isExpired = diffMs <= 0;

  return (
    <div
      className={`flex items-center justify-between px-3 py-1.5 rounded text-sm w-full ${
        isActive && !gameOver
          ? 'bg-gray-700 ring-1 ring-green-500'
          : 'bg-gray-800'
      }`}
    >
      <span className="text-xs text-gray-400">{label}</span>
      {isActive ? (
        <span
          className={`font-bold text-lg ${
            isExpired
              ? 'text-red-500'
              : isUrgent
              ? 'text-red-400 animate-pulse'
              : 'text-white'
          }`}
        >
          {display}
        </span>
      ) : (
        <span className="text-gray-500 text-sm">Waiting</span>
      )}
    </div>
  );
}

interface GameClockProps {
  timeRemaining: [number, number];
  currentPlayer: number;
  myColor: number;
  gameOver: boolean;
  /** Override labels for each player [player0Label, player1Label] */
  labels?: [string, string];
}

export default function GameClock({
  timeRemaining,
  currentPlayer,
  myColor,
  gameOver,
  labels,
}: GameClockProps) {
  const opponentColor = 1 - myColor;

  const getLabel = (player: number) => {
    if (labels) return labels[player];
    return player === myColor ? 'You' : (player === 0 ? 'Blue' : 'Red');
  };

  return (
    <div className="flex items-center gap-2">
      <PlayerClock
        time={timeRemaining[opponentColor]}
        label={getLabel(opponentColor)}
        isActive={currentPlayer === opponentColor}
        gameOver={gameOver}
      />
      <PlayerClock
        time={timeRemaining[myColor]}
        label={getLabel(myColor)}
        isActive={currentPlayer === myColor}
        gameOver={gameOver}
      />
    </div>
  );
}

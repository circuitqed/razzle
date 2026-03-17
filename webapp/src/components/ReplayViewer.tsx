import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import Board from './Board';
import ReplayControls from './ReplayControls';
import MoveHistory from './MoveHistory';
import * as gamesApi from '../api/games';
import type { GameFull } from '../api/games';
import { decodeMove } from '../types';
import { reconstructPositions, getLastMoveAtPosition, type ReplayState } from '../utils/replay';
import { exportGameGif } from '../utils/exportGif';

const END_TURN_MOVE = -1;
const REPLAY_ANIMATION_MS = 500; // slower per-segment for replay visibility

interface ReplayViewerProps {
  gameId: string;
  onClose: () => void;
}

export default function ReplayViewer({ gameId, onClose }: ReplayViewerProps) {
  const [gameData, setGameData] = useState<GameFull | null>(null);
  const [positions, setPositions] = useState<ReplayState[]>([]);
  const [currentPly, setCurrentPly] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [gifProgress, setGifProgress] = useState<number | null>(null); // null = not exporting
  // Server-side analysis removed to avoid server load.
  // TODO: re-add using client-side AI (Web Worker + ONNX)

  // Load game data
  useEffect(() => {
    const loadGame = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await gamesApi.getGameFull(gameId);
        setGameData(data);
        const pos = reconstructPositions(data.moves);
        setPositions(pos);
        setCurrentPly(0);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load game');
      } finally {
        setIsLoading(false);
      }
    };
    loadGame();
  }, [gameId]);

  // Compute turn boundaries: plies where the current player changes
  // These are the "interesting" positions to step between
  const turnBoundaries = useMemo(() => {
    if (positions.length === 0) return [0];
    const boundaries = [0];
    for (let i = 1; i < positions.length; i++) {
      if (positions[i].currentPlayer !== positions[i - 1].currentPlayer) {
        boundaries.push(i);
      }
    }
    // Always include the final position
    const lastPly = positions.length - 1;
    if (boundaries[boundaries.length - 1] !== lastPly) {
      boundaries.push(lastPly);
    }
    return boundaries;
  }, [positions]);

  // Auto-play timer (steps by complete turns, paced for animation)
  useEffect(() => {
    if (!isPlaying || !gameData) return;

    // Compute how many non-END_TURN moves are in the current turn step
    const nextBoundary = turnBoundaries.find(b => b > currentPly);
    const numMoves = nextBoundary !== undefined
      ? gameData.moves.slice(currentPly, nextBoundary).filter(m => m !== END_TURN_MOVE).length
      : 1;
    // Give enough time for animation + a short pause
    const animTime = Math.max(numMoves, 1) * REPLAY_ANIMATION_MS + 200;
    const delay = Math.max(animTime, 1000) / playSpeed;

    const timeout = setTimeout(() => {
      setCurrentPly((ply) => {
        const next = turnBoundaries.find(b => b > ply);
        if (next === undefined) {
          setIsPlaying(false);
          return ply;
        }
        return next;
      });
    }, delay);

    return () => clearTimeout(timeout);
  }, [isPlaying, gameData, playSpeed, turnBoundaries, currentPly]);

  const handleFirst = useCallback(() => {
    setCurrentPly(0);
    setIsPlaying(false);
  }, []);

  const handlePrevious = useCallback(() => {
    setCurrentPly((ply) => {
      // Find the last boundary strictly before current ply
      for (let i = turnBoundaries.length - 1; i >= 0; i--) {
        if (turnBoundaries[i] < ply) return turnBoundaries[i];
      }
      return 0;
    });
  }, [turnBoundaries]);

  const handleNext = useCallback(() => {
    setCurrentPly((ply) => {
      // Find the first boundary strictly after current ply
      const next = turnBoundaries.find(b => b > ply);
      return next !== undefined ? next : ply;
    });
  }, [turnBoundaries]);

  const handleLast = useCallback(() => {
    if (turnBoundaries.length === 0) return;
    setCurrentPly(turnBoundaries[turnBoundaries.length - 1]);
    setIsPlaying(false);
  }, [turnBoundaries]);

  const handleTogglePlay = useCallback(() => {
    if (turnBoundaries.length === 0) return;
    const lastBoundary = turnBoundaries[turnBoundaries.length - 1];
    if (currentPly >= lastBoundary) {
      setCurrentPly(0);
    }
    setIsPlaying((p) => !p);
  }, [turnBoundaries, currentPly]);

  // Seek by turn index (from slider)
  const handleSeekTurn = useCallback((turnIndex: number) => {
    const ply = turnBoundaries[turnIndex] ?? 0;
    setCurrentPly(ply);
    setIsPlaying(false);
  }, [turnBoundaries]);

  const handleExportGif = useCallback(async () => {
    if (positions.length === 0 || gifProgress !== null) return;
    setGifProgress(0);
    try {
      const blob = await exportGameGif(positions, turnBoundaries, setGifProgress);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `game-${gameId}.gif`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('GIF export failed:', err);
    } finally {
      setGifProgress(null);
    }
  }, [positions, turnBoundaries, gameId, gifProgress]);

  // Track ply direction for animation
  const prevPlyRef = useRef(0);
  const prevPly = prevPlyRef.current;
  prevPlyRef.current = currentPly;
  const direction = currentPly > prevPly ? 'forward' : currentPly < prevPly ? 'backward' : 'same';

  // Current position
  const currentState = positions[currentPly];

  // Turn-based indices for display (slider, "Turn X / Y")
  const totalTurns = turnBoundaries.length - 1;
  const currentTurnIndex = useMemo(() => {
    for (let i = turnBoundaries.length - 1; i >= 0; i--) {
      if (turnBoundaries[i] <= currentPly) return i;
    }
    return 0;
  }, [turnBoundaries, currentPly]);

  // Animation: only animate single-turn steps (Next/Previous), not multi-turn jumps (First/Last/slider)
  let lastMove: { from: number; to: number } | null = null;
  let lastTurnMoves: { from: number; to: number }[] | undefined = undefined;

  // Check if this is a single-turn step by comparing turn boundary indices
  const prevTurnIndex = (() => {
    for (let i = turnBoundaries.length - 1; i >= 0; i--) {
      if (turnBoundaries[i] <= prevPly) return i;
    }
    return 0;
  })();
  const isSingleTurnStep = Math.abs(currentTurnIndex - prevTurnIndex) === 1;

  if (gameData && direction === 'forward' && currentPly > 0 && isSingleTurnStep) {
    // Extract all moves in this turn step (prevPly..currentPly), excluding END_TURN
    const turnSlice = gameData.moves.slice(prevPly, currentPly);
    const realMoves = turnSlice
      .filter(m => m !== END_TURN_MOVE)
      .map(m => {
        const { src, dst } = decodeMove(m);
        return { from: src, to: dst };
      });

    if (realMoves.length > 1) {
      // Multi-pass: use lastTurnMoves for waypoint animation
      lastTurnMoves = realMoves;
    } else {
      // Single move (knight or single pass)
      lastMove = getLastMoveAtPosition(gameData.moves, currentPly);
    }
  } else if (gameData && direction === 'backward' && prevPly > 0 && isSingleTurnStep) {
    const undoneMove = getLastMoveAtPosition(gameData.moves, prevPly);
    if (undoneMove) {
      lastMove = { from: undoneMove.to, to: undoneMove.from };
    }
  }

  // Get moves up to current ply for display
  const movesUpToPly = useMemo(() => {
    if (!gameData) return [];
    return gameData.moves.slice(0, currentPly);
  }, [gameData, currentPly]);

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg p-8">
          <div className="text-white">Loading game...</div>
        </div>
      </div>
    );
  }

  if (error || !gameData || !currentState) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg p-8">
          <div className="text-red-400 mb-4">{error || 'Failed to load game'}</div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg p-4 max-w-4xl w-full mx-4 max-h-[95vh] overflow-y-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-xl font-bold text-white">Game Replay</h2>
            <p className="text-sm text-gray-400">
              {gameData.player2_type === 'ai' ? 'vs AI' : '2 Player'} • {' '}
              {gameData.status === 'finished'
                ? gameData.winner === 0
                  ? 'Blue Won'
                  : gameData.winner === 1
                  ? 'Red Won'
                  : 'Draw'
                : 'In Progress'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {gifProgress !== null ? (
              <span className="text-sm text-gray-400">
                Generating GIF... {Math.round(gifProgress * 100)}%
              </span>
            ) : (
              <button
                onClick={handleExportGif}
                className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors"
                title="Export as animated GIF"
              >
                Export GIF
              </button>
            )}
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Main content */}
        <div className="flex flex-col lg:flex-row gap-4 items-center lg:items-start">
          {/* Board */}
          <div>
            <Board
              board={currentState.board}
              currentPlayer={currentState.currentPlayer as 0 | 1}
              legalMoves={[]}
              selectedSquare={null}
              onSquareClick={() => {}}
              touchedMask={currentState.touchedMask}
              lastMove={lastMove}
              lastTurnMoves={lastTurnMoves}
              animationDuration={REPLAY_ANIMATION_MS}
            />
          </div>

          {/* Side panel */}
          <div className="flex flex-col gap-4 w-full lg:w-auto">
            {/* Move History */}
            <div className="hidden lg:block">
              <MoveHistory moves={movesUpToPly} />
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="mt-4 flex justify-center">
          <ReplayControls
            currentTurn={currentTurnIndex}
            totalTurns={totalTurns}
            isPlaying={isPlaying}
            playSpeed={playSpeed}
            onFirst={handleFirst}
            onPrevious={handlePrevious}
            onNext={handleNext}
            onLast={handleLast}
            onTogglePlay={handleTogglePlay}
            onSeek={handleSeekTurn}
            onSpeedChange={setPlaySpeed}
          />
        </div>

        {/* Keyboard hints */}
        <div className="mt-4 text-center text-xs text-gray-500">
          Arrow keys: navigate • Space: play/pause • Home/End: first/last
        </div>
      </div>
    </div>
  );
}

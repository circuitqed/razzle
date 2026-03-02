import { useState, useEffect, useCallback, useMemo } from 'react';
import Board from './Board';
import ReplayControls from './ReplayControls';
import MoveHistory from './MoveHistory';
import * as gamesApi from '../api/games';
import type { GameFull } from '../api/games';
import { reconstructPositions, getLastMoveAtPosition, type ReplayState } from '../utils/replay';

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

  // Auto-play timer (steps by complete turns)
  useEffect(() => {
    if (!isPlaying || !gameData) return;

    const interval = setInterval(() => {
      setCurrentPly((ply) => {
        const nextBoundary = turnBoundaries.find(b => b > ply);
        if (nextBoundary === undefined) {
          setIsPlaying(false);
          return ply;
        }
        return nextBoundary;
      });
    }, 1000 / playSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, gameData, playSpeed, turnBoundaries]);

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
    if (!gameData) return;
    setCurrentPly(gameData.moves.length);
    setIsPlaying(false);
  }, [gameData]);

  const handleTogglePlay = useCallback(() => {
    if (!gameData) return;
    if (currentPly >= gameData.moves.length) {
      setCurrentPly(0);
    }
    setIsPlaying((p) => !p);
  }, [gameData, currentPly]);

  const handleSeek = useCallback((ply: number) => {
    setCurrentPly(ply);
    setIsPlaying(false);
  }, []);

  // Current position
  const currentState = positions[currentPly];
  const lastMove = currentPly > 0 && gameData
    ? getLastMoveAtPosition(gameData.moves, currentPly)
    : null;

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
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
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
            currentPly={currentPly}
            maxPly={gameData.moves.length}
            isPlaying={isPlaying}
            playSpeed={playSpeed}
            onFirst={handleFirst}
            onPrevious={handlePrevious}
            onNext={handleNext}
            onLast={handleLast}
            onTogglePlay={handleTogglePlay}
            onSeek={handleSeek}
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

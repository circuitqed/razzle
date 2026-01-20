import { useState, useEffect, useCallback, useMemo } from 'react';
import Board from './Board';
import ReplayControls from './ReplayControls';
import MoveHistory from './MoveHistory';
import * as gamesApi from '../api/games';
import type { GameFull, MoveClassification } from '../api/games';
import { reconstructPositions, getLastMoveAtPosition, type ReplayState } from '../utils/replay';
import { decodeMove, squareToAlgebraic } from '../types';

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
  const [analysis, setAnalysis] = useState<MoveClassification[] | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

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

  // Auto-play timer
  useEffect(() => {
    if (!isPlaying || !gameData) return;

    const interval = setInterval(() => {
      setCurrentPly((ply) => {
        if (ply >= gameData.moves.length) {
          setIsPlaying(false);
          return ply;
        }
        return ply + 1;
      });
    }, 1000 / playSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, gameData, playSpeed]);

  const handleFirst = useCallback(() => {
    setCurrentPly(0);
    setIsPlaying(false);
  }, []);

  const handlePrevious = useCallback(() => {
    setCurrentPly((ply) => Math.max(0, ply - 1));
  }, []);

  const handleNext = useCallback(() => {
    if (!gameData) return;
    setCurrentPly((ply) => Math.min(gameData.moves.length, ply + 1));
  }, [gameData]);

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

  const handleAnalyze = useCallback(async () => {
    if (!gameData) return;
    setIsAnalyzing(true);
    try {
      const result = await gamesApi.analyzeGame(gameId, 100);
      setAnalysis(result.move_analyses);
    } catch (err) {
      console.error('Analysis failed:', err);
    } finally {
      setIsAnalyzing(false);
    }
  }, [gameData, gameId]);

  // Current position
  const currentState = positions[currentPly];
  const lastMove = currentPly > 0 && gameData
    ? getLastMoveAtPosition(gameData.moves, currentPly)
    : null;

  // Convert moves to move history format
  const moveHistory = useMemo(() => {
    if (!gameData) return [];
    return gameData.moves.slice(0, currentPly).map((move, idx) => {
      if (move === -1) {
        return {
          move,
          algebraic: 'End Turn',
          player: (idx % 2 === 0 ? 0 : 1) as 0 | 1,
        };
      }
      const { src, dst } = decodeMove(move);
      const player = positions[idx]?.currentPlayer;
      return {
        move,
        algebraic: `${squareToAlgebraic(src)}-${squareToAlgebraic(dst)}`,
        player: (player === 0 || player === 1 ? player : 0) as 0 | 1,
      };
    });
  }, [gameData, currentPly, positions]);

  // Get analysis for current move
  const currentMoveAnalysis = currentPly > 0 && analysis
    ? analysis[currentPly - 1]
    : null;

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
            {/* Move analysis badge */}
            {currentMoveAnalysis && (
              <div className={`px-3 py-2 rounded text-sm ${
                currentMoveAnalysis.classification === 'best' ? 'bg-green-900 text-green-300' :
                currentMoveAnalysis.classification === 'good' ? 'bg-blue-900 text-blue-300' :
                currentMoveAnalysis.classification === 'inaccuracy' ? 'bg-yellow-900 text-yellow-300' :
                currentMoveAnalysis.classification === 'mistake' ? 'bg-orange-900 text-orange-300' :
                'bg-red-900 text-red-300'
              }`}>
                <span className="font-medium capitalize">{currentMoveAnalysis.classification}</span>
                <span className="text-xs ml-2">
                  ({currentMoveAnalysis.algebraic}, eval: {currentMoveAnalysis.value_after.toFixed(2)})
                </span>
                {currentMoveAnalysis.classification !== 'best' && (
                  <div className="text-xs mt-1 opacity-80">
                    Best: {currentMoveAnalysis.best_move_algebraic}
                  </div>
                )}
              </div>
            )}

            {/* Move History */}
            <div className="hidden lg:block">
              <MoveHistory moves={moveHistory} />
            </div>

            {/* Analyze button */}
            {!analysis && (
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze Game'}
              </button>
            )}

            {/* Analysis summary */}
            {analysis && (
              <div className="bg-gray-800 rounded p-3 text-sm">
                <h4 className="font-medium text-gray-300 mb-2">Analysis Summary</h4>
                <div className="grid grid-cols-5 gap-1 text-xs">
                  <div className="text-center">
                    <div className="text-green-400 font-medium">{analysis.filter(m => m.classification === 'best').length}</div>
                    <div className="text-gray-500">Best</div>
                  </div>
                  <div className="text-center">
                    <div className="text-blue-400 font-medium">{analysis.filter(m => m.classification === 'good').length}</div>
                    <div className="text-gray-500">Good</div>
                  </div>
                  <div className="text-center">
                    <div className="text-yellow-400 font-medium">{analysis.filter(m => m.classification === 'inaccuracy').length}</div>
                    <div className="text-gray-500">Inac.</div>
                  </div>
                  <div className="text-center">
                    <div className="text-orange-400 font-medium">{analysis.filter(m => m.classification === 'mistake').length}</div>
                    <div className="text-gray-500">Mist.</div>
                  </div>
                  <div className="text-center">
                    <div className="text-red-400 font-medium">{analysis.filter(m => m.classification === 'blunder').length}</div>
                    <div className="text-gray-500">Blun.</div>
                  </div>
                </div>
              </div>
            )}
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

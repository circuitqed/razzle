import { useState, useCallback } from 'react';
import Board from './Board';
import AnalysisSidebar from './AnalysisSidebar';
import * as gamesApi from '../api/games';
import type { MoveAnalysis } from '../api/games';
import { BoardState } from '../types';
import { getInitialState } from '../utils/replay';

interface AnalysisBoardProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AnalysisBoard({ isOpen, onClose }: AnalysisBoardProps) {
  const initialState = getInitialState();

  const [board, setBoard] = useState<BoardState>(initialState.board);
  const [currentPlayer, setCurrentPlayer] = useState<0 | 1>(0);
  const [touchedMask, setTouchedMask] = useState('0');
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [isEditMode, setIsEditMode] = useState(true);

  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisValue, setAnalysisValue] = useState(0);
  const [topMoves, setTopMoves] = useState<MoveAnalysis[]>([]);
  const [legalMoves, setLegalMoves] = useState<number[]>([]);

  // Check what's on a square (use BigInt for 56-bit board)
  const getPieceAt = useCallback((square: number, boardState: BoardState) => {
    const mask = BigInt(1) << BigInt(square);
    const p1Pieces = BigInt(boardState.p1_pieces);
    const p2Pieces = BigInt(boardState.p2_pieces);
    const p1Ball = BigInt(boardState.p1_ball);
    const p2Ball = BigInt(boardState.p2_ball);

    if (p1Pieces & mask) {
      return { player: 0 as const, hasBall: !!(p1Ball & mask) };
    }
    if (p2Pieces & mask) {
      return { player: 1 as const, hasBall: !!(p2Ball & mask) };
    }
    return null;
  }, []);

  const handleSquareClick = useCallback((square: number) => {
    if (!isEditMode) return;

    const clickedPiece = getPieceAt(square, board);

    if (selectedSquare === null) {
      // Nothing selected - select a piece if there is one
      if (clickedPiece) {
        setSelectedSquare(square);
      }
    } else {
      // Something is selected
      if (square === selectedSquare) {
        // Clicked same square - deselect
        setSelectedSquare(null);
      } else if (clickedPiece) {
        // Clicked another piece - select that one instead
        setSelectedSquare(square);
      } else {
        // Clicked empty square - move the selected piece there
        const selectedPiece = getPieceAt(selectedSquare, board);
        if (selectedPiece) {
          setBoard(prev => {
            const srcMask = BigInt(1) << BigInt(selectedSquare);
            const dstMask = BigInt(1) << BigInt(square);

            // Convert to BigInt for operations
            let p1Pieces = BigInt(prev.p1_pieces);
            let p1Ball = BigInt(prev.p1_ball);
            let p2Pieces = BigInt(prev.p2_pieces);
            let p2Ball = BigInt(prev.p2_ball);

            // Move piece from source to destination
            if (selectedPiece.player === 0) {
              p1Pieces = (p1Pieces & ~srcMask) | dstMask;
              if (selectedPiece.hasBall) {
                p1Ball = (p1Ball & ~srcMask) | dstMask;
              }
            } else {
              p2Pieces = (p2Pieces & ~srcMask) | dstMask;
              if (selectedPiece.hasBall) {
                p2Ball = (p2Ball & ~srcMask) | dstMask;
              }
            }

            // Update touched mask if the piece was touched
            const currentTouched = BigInt(touchedMask);
            if (currentTouched & srcMask) {
              setTouchedMask(String((currentTouched & ~srcMask) | dstMask));
            }

            return {
              p1_pieces: String(p1Pieces),
              p1_ball: String(p1Ball),
              p2_pieces: String(p2Pieces),
              p2_ball: String(p2Ball),
            };
          });

          // Clear analysis when board changes
          setTopMoves([]);
          setAnalysisValue(0);
          setLegalMoves([]);
        }
        setSelectedSquare(null);
      }
    }
  }, [isEditMode, selectedSquare, board, getPieceAt, touchedMask]);

  const handleSquareDoubleClick = useCallback((square: number) => {
    if (!isEditMode) return;

    const piece = getPieceAt(square, board);
    if (piece && !piece.hasBall) {
      // Toggle touched status for non-ball pieces
      const mask = BigInt(1) << BigInt(square);
      setTouchedMask(prev => String(BigInt(prev) ^ mask));
    }
  }, [isEditMode, board, getPieceAt]);

  const handleReset = useCallback(() => {
    const initial = getInitialState();
    setBoard(initial.board);
    setCurrentPlayer(0);
    setTouchedMask('0');
    setSelectedSquare(null);
    setTopMoves([]);
    setAnalysisValue(0);
    setLegalMoves([]);
  }, []);

  const handleAnalyze = useCallback(async () => {
    setIsAnalyzing(true);
    try {
      const result = await gamesApi.analyzePosition({
        pieces: [board.p1_pieces, board.p2_pieces],
        balls: [board.p1_ball, board.p2_ball],
        current_player: currentPlayer,
        touched_mask: touchedMask,
        has_passed: false,
        simulations: 400,
      });

      setAnalysisValue(result.value);
      setTopMoves(result.top_moves);
      setLegalMoves(result.legal_moves);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, [board, currentPlayer, touchedMask]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg p-4 max-w-4xl w-full mx-4 max-h-[95vh] overflow-y-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-xl font-bold text-white">Position Editor</h2>
            <p className="text-sm text-gray-400">
              Set up any position and analyze it with the AI
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

        {/* Mode toggle and controls */}
        <div className="flex flex-wrap gap-2 mb-4 items-center">
          <button
            onClick={() => setIsEditMode(true)}
            className={`px-4 py-2 rounded font-medium transition-colors ${
              isEditMode
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Edit Position
          </button>
          <button
            onClick={() => setIsEditMode(false)}
            className={`px-4 py-2 rounded font-medium transition-colors ${
              !isEditMode
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Analyze
          </button>

          <div className="w-px h-8 bg-gray-700 mx-2" />

          {/* Current player toggle */}
          <span className="text-sm text-gray-400">To move:</span>
          <button
            onClick={() => setCurrentPlayer(0)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              currentPlayer === 0
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Blue
          </button>
          <button
            onClick={() => setCurrentPlayer(1)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              currentPlayer === 1
                ? 'bg-red-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Red
          </button>

          <div className="flex-1" />

          <button
            onClick={handleReset}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded font-medium transition-colors"
          >
            Reset
          </button>

          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze Position'}
          </button>
        </div>

        {/* Main content */}
        <div className="flex flex-col lg:flex-row gap-4 items-start">
          {/* Board */}
          <div>
            <Board
              board={board}
              currentPlayer={currentPlayer}
              legalMoves={isEditMode ? [] : legalMoves}
              selectedSquare={selectedSquare}
              onSquareClick={handleSquareClick}
              onSquareDoubleClick={handleSquareDoubleClick}
              touchedMask={touchedMask}
            />
          </div>

          {/* Side panel */}
          <div className="flex-1 min-w-0">
            {isEditMode ? (
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="font-medium text-gray-300 mb-3">Instructions</h3>
                <ul className="text-sm text-gray-400 space-y-2">
                  <li>• Click a piece to select it, then click an empty square to move it</li>
                  <li>• Double-click a piece (without ball) to toggle "touched" status</li>
                  <li>• Touched pieces are highlighted and cannot receive passes</li>
                  <li>• Select which player is to move using the buttons above</li>
                </ul>

                {touchedMask !== '0' && (
                  <div className="mt-4 pt-4 border-t border-gray-700">
                    <p className="text-xs text-gray-500">
                      Touched mask: {touchedMask} ({BigInt(touchedMask).toString(2).split('1').length - 1} pieces)
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <AnalysisSidebar
                value={analysisValue}
                currentPlayer={currentPlayer}
                topMoves={topMoves}
                isAnalyzing={isAnalyzing}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

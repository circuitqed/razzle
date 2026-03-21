import { useMemo } from 'react';
import Board from './Board';
import { useTutorial } from '../hooks/useTutorial';
import type { TutorialStep as HookStep } from '../hooks/useTutorial';
import { TUTORIAL_STEPS } from '../data/tutorialSteps';
import { mustPass as checkMustPass } from '../engine/moves';

interface TutorialProps {
  onComplete: () => void;
  onSkip: () => void;
}

export default function Tutorial({ onComplete, onSkip }: TutorialProps) {
  const hookSteps = useMemo<HookStep[]>(
    () =>
      TUTORIAL_STEPS.map((s) => ({
        title: s.title,
        instruction: s.instruction,
        hint: s.hint,
        state: s.boardState,
        allowedMoves: s.allowedMoves,
        requireEndTurn: s.requireEndTurn,
        nextStepState: s.nextStepState,
        chainMoves: s.chainMoves,
        highlightSquares: s.highlightSquares,
        preState: s.preState,
        preMessage: s.preMessage,
        completionMessage: s.completionMessage,
        chainPreState: s.chainPreState,
        chainPreMessage: s.chainPreMessage,
        preMove: s.preMove,
        chainPreMove: s.chainPreMove,
        suggestedMoves: s.suggestedMoves,
      })),
    [],
  );

  const tut = useTutorial({ steps: hookSteps, onComplete, onSkip });

  if (!tut.gameState) return null;

  const isLastStep = tut.currentStep === tut.totalSteps - 1;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex flex-col items-center justify-center p-4">
      {/* Instruction panel — fixed height to prevent board shifting */}
      <div className="max-w-md w-full text-center mb-3 h-20 flex flex-col justify-center">
        <h2 className="text-lg sm:text-xl font-bold text-white mb-0.5">
          {tut.stepTitle}
        </h2>
        <p className="text-gray-300 text-sm leading-tight transition-opacity duration-300">
          {tut.showingPreState ? tut.preMessage : tut.stepInstruction}
        </p>
        {!tut.stepComplete && !tut.showingPreState && (
          <p className="text-gray-500 text-xs mt-0.5 italic">{tut.stepHint}</p>
        )}
      </div>

      {/* Board */}
      <div className="flex-shrink-0">
        <Board
          board={tut.gameState.board}
          currentPlayer={tut.gameState.current_player}
          legalMoves={tut.showingPreState ? [] : tut.gameState.legal_moves}
          selectedSquare={tut.selectedSquare}
          onSquareClick={tut.handleSquareClick}
          onDragMove={tut.handleDragMove}
          flipped={false}
          touchedMask={tut.gameState.touched_mask}
          mustPass={checkMustPass(tut.engineState)}
          lastMove={tut.preAnimMove}
          suggestedMoves={tut.suggestedMoves}
        />
      </div>

      {/* Controls — fixed height to prevent board shifting */}
      <div className="max-w-md w-full flex flex-col items-center mt-3 h-24 justify-start gap-2">
        {/* Complete Pass button */}
        {tut.canEndTurn && !tut.stepComplete && (
          <button
            onClick={tut.endTurn}
            className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium text-white transition-colors animate-pulse"
          >
            Complete Pass
          </button>
        )}

        {/* Completion message + Next button */}
        {tut.stepComplete && (
          <>
            {tut.completionMessage && (
              <p className="text-green-400 font-medium">{tut.completionMessage}</p>
            )}
            <button
              onClick={tut.nextStep}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium text-white transition-colors"
            >
              {isLastStep ? 'Start Playing!' : 'Next'}
            </button>
          </>
        )}

        {/* Progress dots + skip — always visible at bottom */}
        <div className="flex items-center gap-2 mt-auto">
          {Array.from({ length: tut.totalSteps }, (_, i) => (
            <div
              key={i}
              className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                i === tut.currentStep
                  ? 'bg-blue-400'
                  : i < tut.currentStep
                    ? 'bg-blue-700'
                    : 'bg-gray-600'
              }`}
            />
          ))}
        </div>
        <button
          onClick={tut.skipTutorial}
          className="text-gray-500 hover:text-gray-300 text-xs transition-colors"
        >
          Skip Tutorial
        </button>
      </div>
    </div>
  );
}

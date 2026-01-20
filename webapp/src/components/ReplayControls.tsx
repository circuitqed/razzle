import { useEffect } from 'react';

interface ReplayControlsProps {
  currentPly: number;
  maxPly: number;
  isPlaying: boolean;
  playSpeed: number;
  onFirst: () => void;
  onPrevious: () => void;
  onNext: () => void;
  onLast: () => void;
  onTogglePlay: () => void;
  onSeek: (ply: number) => void;
  onSpeedChange: (speed: number) => void;
}

export default function ReplayControls({
  currentPly,
  maxPly,
  isPlaying,
  playSpeed,
  onFirst,
  onPrevious,
  onNext,
  onLast,
  onTogglePlay,
  onSeek,
  onSpeedChange,
}: ReplayControlsProps) {
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      switch (e.key) {
        case 'ArrowLeft':
          onPrevious();
          break;
        case 'ArrowRight':
          onNext();
          break;
        case 'Home':
          onFirst();
          break;
        case 'End':
          onLast();
          break;
        case ' ':
          e.preventDefault();
          onTogglePlay();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onFirst, onPrevious, onNext, onLast, onTogglePlay]);

  return (
    <div className="bg-gray-800 rounded-lg p-4 w-full max-w-md">
      {/* Navigation buttons */}
      <div className="flex justify-center items-center gap-2 mb-3">
        <button
          onClick={onFirst}
          disabled={currentPly === 0}
          className="p-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded transition-colors"
          title="First position (Home)"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M15.707 15.707a1 1 0 01-1.414 0l-5-5a1 1 0 010-1.414l5-5a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 010 1.414zm-6 0a1 1 0 01-1.414 0l-5-5a1 1 0 010-1.414l5-5a1 1 0 011.414 1.414L5.414 10l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
          </svg>
        </button>

        <button
          onClick={onPrevious}
          disabled={currentPly === 0}
          className="p-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded transition-colors"
          title="Previous move (Left Arrow)"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        </button>

        <button
          onClick={onTogglePlay}
          className="p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-full transition-colors"
          title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
        >
          {isPlaying ? (
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
            </svg>
          )}
        </button>

        <button
          onClick={onNext}
          disabled={currentPly === maxPly}
          className="p-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded transition-colors"
          title="Next move (Right Arrow)"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
          </svg>
        </button>

        <button
          onClick={onLast}
          disabled={currentPly === maxPly}
          className="p-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded transition-colors"
          title="Last position (End)"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10.293 15.707a1 1 0 010-1.414L14.586 10l-4.293-4.293a1 1 0 111.414-1.414l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0z" clipRule="evenodd" />
            <path fillRule="evenodd" d="M4.293 15.707a1 1 0 010-1.414L8.586 10 4.293 5.707a1 1 0 011.414-1.414l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0z" clipRule="evenodd" />
          </svg>
        </button>
      </div>

      {/* Slider */}
      <div className="mb-3">
        <input
          type="range"
          min={0}
          max={maxPly}
          value={currentPly}
          onChange={(e) => onSeek(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>Start</span>
          <span>Move {currentPly} / {maxPly}</span>
          <span>End</span>
        </div>
      </div>

      {/* Speed control */}
      <div className="flex justify-center items-center gap-2">
        <span className="text-xs text-gray-400">Speed:</span>
        {[0.5, 1, 2, 4].map((speed) => (
          <button
            key={speed}
            onClick={() => onSpeedChange(speed)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              playSpeed === speed
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {speed}x
          </button>
        ))}
      </div>
    </div>
  );
}

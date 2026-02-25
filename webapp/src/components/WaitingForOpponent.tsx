/**
 * Waiting for Opponent component.
 * Displays shareable game code while waiting for opponent to join.
 */

import { useState, useEffect } from 'react';

interface WaitingForOpponentProps {
  joinCode: string;
  hostColor: number;
  onCancel: () => void;
}

export default function WaitingForOpponent({
  joinCode,
  hostColor,
  onCancel,
}: WaitingForOpponentProps) {
  const [copied, setCopied] = useState(false);
  const [dots, setDots] = useState('');

  // Animate loading dots
  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
    }, 500);
    return () => clearInterval(interval);
  }, []);

  const handleCopyCode = async () => {
    try {
      await navigator.clipboard.writeText(joinCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleCopyLink = async () => {
    const url = `${window.location.origin}/join/${joinCode}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleShare = async () => {
    const url = `${window.location.origin}/join/${joinCode}`;
    const text = `Join my Razzle Dazzle game! Code: ${joinCode}`;

    if (navigator.share) {
      try {
        await navigator.share({ title: 'Razzle Dazzle', text, url });
      } catch {
        // User cancelled or share failed
      }
    } else {
      handleCopyLink();
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-8 max-w-md w-full mx-4 text-center">
        {/* Color indicator */}
        <div className="mb-4 flex items-center justify-center gap-2">
          <span
            className={`w-6 h-6 rounded-full ${
              hostColor === 0 ? 'bg-blue-500' : 'bg-red-500'
            }`}
          ></span>
          <span className="text-gray-400">
            You are playing as {hostColor === 0 ? 'Blue' : 'Red'}
          </span>
        </div>

        {/* Waiting message */}
        <h2 className="text-2xl font-bold text-white mb-2">
          Waiting for opponent{dots}
        </h2>
        <p className="text-gray-400 mb-6">
          Share this code with a friend to start the game
        </p>

        {/* Join code */}
        <div className="bg-gray-900 rounded-lg p-6 mb-6">
          <p className="text-gray-500 text-sm mb-2">Game Code</p>
          <p className="text-5xl font-mono font-bold text-white tracking-[0.3em]">
            {joinCode}
          </p>
        </div>

        {/* Action buttons */}
        <div className="flex flex-col gap-3 mb-6">
          <button
            onClick={handleCopyCode}
            className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium transition-colors flex items-center justify-center gap-2"
          >
            {copied ? (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Copied!
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy Code
              </>
            )}
          </button>

          {'share' in navigator && (
            <button
              onClick={handleShare}
              className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded font-medium transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
              </svg>
              Share Link
            </button>
          )}

          <button
            onClick={handleCopyLink}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded font-medium transition-colors text-sm"
          >
            Copy Link: {window.location.origin}/join/{joinCode}
          </button>
        </div>

        {/* Cancel button */}
        <button
          onClick={onCancel}
          className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
        >
          Cancel and leave
        </button>

        {/* Instructions */}
        <div className="mt-6 pt-4 border-t border-gray-700">
          <p className="text-sm text-gray-500">
            The game will start automatically when your opponent joins.
          </p>
        </div>
      </div>
    </div>
  );
}

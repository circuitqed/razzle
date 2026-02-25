import { useState, useCallback } from 'react';
import { submitBugReport } from '../api/engine';

interface BugReportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  gameState?: object | null;
  rawMoves?: number[];
  gameMode?: string;
  aiModel?: string | null;
  gameId?: string | null;
}

export default function BugReportDialog({
  isOpen,
  onClose,
  gameState,
  rawMoves,
  gameMode,
  aiModel,
  gameId,
}: BugReportDialogProps) {
  const [description, setDescription] = useState('');
  const [status, setStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');

  const handleClose = useCallback(() => {
    setDescription('');
    setStatus('idle');
    setErrorMsg('');
    onClose();
  }, [onClose]);

  const handleSubmit = useCallback(async () => {
    if (!description.trim()) return;
    setStatus('submitting');
    try {
      await submitBugReport({
        description: description.trim(),
        game_id: gameId,
        game_state: gameState,
        moves: rawMoves,
        game_mode: gameMode,
        ai_model: aiModel,
        client_info: navigator.userAgent,
      });
      setStatus('success');
      setTimeout(handleClose, 1500);
    } catch (e) {
      setStatus('error');
      setErrorMsg(e instanceof Error ? e.message : 'Failed to submit');
    }
  }, [description, gameId, gameState, rawMoves, gameMode, aiModel, handleClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={handleClose} />

      {/* Dialog */}
      <div className="relative bg-gray-800 rounded-lg shadow-xl max-w-md w-full p-6">
        <h2 className="text-xl font-bold mb-4">Report a bug or request a feature</h2>

        {status === 'success' ? (
          <p className="text-green-400 text-center py-4">Thanks for the report!</p>
        ) : (
          <>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what went wrong..."
              className="w-full h-32 bg-gray-700 text-white rounded p-3 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
              autoFocus
              disabled={status === 'submitting'}
            />

            {status === 'error' && (
              <p className="text-red-400 text-sm mt-2">{errorMsg}</p>
            )}

            <div className="flex justify-end gap-3 mt-4">
              <button
                onClick={handleClose}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors"
                disabled={status === 'submitting'}
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                disabled={!description.trim() || status === 'submitting'}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:text-gray-400 rounded font-medium transition-colors"
              >
                {status === 'submitting' ? 'Sending...' : 'Submit'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

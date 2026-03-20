import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface UsernamePickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  tempToken: string;
  suggestedName: string;
}

export default function UsernamePickerModal({ isOpen, onClose, tempToken, suggestedName }: UsernamePickerModalProps) {
  const { googleComplete } = useAuth();
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState(suggestedName);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setUsername('');
      setDisplayName(suggestedName);
      setError(null);
    }
  }, [isOpen, suggestedName]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (username.length < 3) {
      setError('Username must be at least 3 characters');
      return;
    }
    if (!/^[a-zA-Z0-9_]+$/.test(username)) {
      setError('Username can only contain letters, numbers, and underscores');
      return;
    }

    setIsLoading(true);
    try {
      await googleComplete(tempToken, username, displayName || undefined);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create account');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 max-w-sm w-full mx-4">
        <h2 className="text-xl font-bold text-white mb-2">Choose a Username</h2>
        <p className="text-sm text-gray-400 mb-4">Pick a unique username for your KnightBall account.</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="bg-red-600 text-white px-3 py-2 rounded text-sm">
              {error}
            </div>
          )}

          <div>
            <label className="block text-sm text-gray-300 mb-1">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
              required
              autoFocus
              minLength={3}
              maxLength={32}
              placeholder="e.g. knightmaster42"
            />
            <p className="text-xs text-gray-500 mt-1">3-32 characters, letters, numbers, and underscores only</p>
          </div>

          <div>
            <label className="block text-sm text-gray-300 mb-1">Display Name (optional)</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
              maxLength={64}
            />
          </div>

          <div className="flex gap-2 pt-2">
            <button
              type="submit"
              disabled={isLoading}
              className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
            >
              {isLoading ? 'Creating...' : 'Create Account'}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded font-medium transition-colors"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

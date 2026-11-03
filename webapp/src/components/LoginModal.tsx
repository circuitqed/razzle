import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useDialogA11y } from '../hooks/useDialogA11y';

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSwitchToRegister: () => void;
  onForgotPassword: () => void;
}

export default function LoginModal({ isOpen, onClose, onSwitchToRegister, onForgotPassword }: LoginModalProps) {
  const dialogRef = useDialogA11y(onClose);
  const { loginWithEmail, requestMagicLink, verifyMagicCode, isAuthenticated } = useAuth();
  const [emailOrUsername, setEmailOrUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [magicSent, setMagicSent] = useState(false);
  const [magicLoading, setMagicLoading] = useState(false);
  const [code, setCode] = useState('');

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen) {
      setEmailOrUsername('');
      setPassword('');
      setError(null);
      setMagicSent(false);
      setMagicLoading(false);
      setCode('');
    }
  }, [isOpen]);

  const handleMagicLink = async () => {
    setError(null);
    if (!emailOrUsername.includes('@')) {
      setError('Enter your email address above to get a sign-in link.');
      return;
    }
    setMagicLoading(true);
    try {
      await requestMagicLink(emailOrUsername);
      setMagicSent(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not send the link.');
    } finally {
      setMagicLoading(false);
    }
  };

  // The code is the reliable path in the app: on a fresh install, tapping the
  // email link can open Mail's own browser and sign that in instead.
  const handleCode = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setMagicLoading(true);
    try {
      await verifyMagicCode(emailOrUsername.trim(), code);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not sign in with that code.');
    } finally {
      setMagicLoading(false);
    }
  };

  // Close when sign-in completes outside the form — the native Google flow
  // finishes via a deep link (Safari round-trip), not through this modal.
  useEffect(() => {
    if (isOpen && isAuthenticated) onClose();
  }, [isOpen, isAuthenticated, onClose]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      await loginWithEmail(emailOrUsername, password);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 max-w-sm w-full mx-4">
        <h2 className="text-xl font-bold text-white mb-4">Login</h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="bg-red-600 text-white px-3 py-2 rounded text-sm">
              {error}
            </div>
          )}

          <div>
            <label className="block text-sm text-gray-300 mb-1">Email or Username</label>
            <input
              type="text"
              value={emailOrUsername}
              onChange={(e) => setEmailOrUsername(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
              required
              autoFocus
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-sm text-gray-300">Password</label>
              <button
                type="button"
                onClick={onForgotPassword}
                className="text-xs text-blue-400 hover:text-blue-300"
              >
                Forgot password?
              </button>
            </div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
              required
            />
          </div>

          <div className="flex gap-2 pt-2">
            <button
              type="submit"
              disabled={isLoading}
              className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
            >
              {isLoading ? 'Logging in...' : 'Login'}
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

        {/* Passwordless alternative — works the same on web and in the app */}
        <div className="mt-4 pt-4 border-t border-gray-700">
          {magicSent ? (
            <form onSubmit={handleCode} className="space-y-2">
              <p className="text-sm text-green-300 text-center">
                We emailed you a 6-digit code. Enter it here, or tap the link in the email.
              </p>
              <div className="flex gap-2">
                <input
                  value={code}
                  onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  placeholder="123456"
                  aria-label="Sign-in code"
                  className="flex-1 min-w-0 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-center tracking-[0.3em] focus:outline-none focus:border-blue-500"
                />
                <button
                  type="submit"
                  disabled={code.length !== 6 || magicLoading}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
                >
                  {magicLoading ? '…' : 'Sign in'}
                </button>
              </div>
            </form>
          ) : (
            <button
              type="button"
              onClick={handleMagicLink}
              disabled={magicLoading}
              className="w-full px-4 py-2 text-sm bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 text-white rounded transition-colors"
            >
              {magicLoading ? 'Sending…' : 'Email me a sign-in code instead'}
            </button>
          )}
        </div>

        <div className="mt-4 text-center text-sm text-gray-400">
          Don't have an account?{' '}
          <button
            onClick={onSwitchToRegister}
            className="text-blue-400 hover:text-blue-300"
          >
            Register
          </button>
        </div>
      </div>
    </div>
  );
}

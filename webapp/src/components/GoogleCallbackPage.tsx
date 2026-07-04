import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function GoogleCallbackPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { googleAuth, googleComplete } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const [needsUsername, setNeedsUsername] = useState<{
    tempToken: string;
    suggestedName: string;
    email: string;
  } | null>(null);
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const code = searchParams.get('code');
  const authError = searchParams.get('error');
  // 'native.<nonce>' marks a flow started from the iOS app; the server then
  // returns a one-time app_ticket we hand back via the knightball:// deep link.
  const oauthState = searchParams.get('state');
  const [appReturn, setAppReturn] = useState<string | null>(null);

  const returnToApp = (ticket: string) => {
    const deepLink = `knightball://auth?ticket=${encodeURIComponent(ticket)}`;
    setAppReturn(deepLink);
    window.location.href = deepLink;
  };

  useEffect(() => {
    if (authError) {
      setError(`Google sign-in failed: ${authError}`);
      return;
    }
    if (!code) {
      setError('Missing authorization code');
      return;
    }

    // Send the auth code to our backend
    const doAuth = async () => {
      setIsLoading(true);
      try {
        const result = await googleAuth(code, oauthState);
        if (result.status === 'logged_in') {
          if (result.app_ticket) {
            returnToApp(result.app_ticket);
            return;
          }
          navigate('/', { replace: true });
        } else if (result.status === 'needs_username') {
          setNeedsUsername({
            tempToken: result.temp_token!,
            suggestedName: result.suggested_name || '',
            email: result.email || '',
          });
          setDisplayName(result.suggested_name || '');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Google sign-in failed');
      } finally {
        setIsLoading(false);
      }
    };

    doAuth();
  }, [code, authError]);

  const handleCompleteSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!needsUsername) return;
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
      const result = await googleComplete(needsUsername.tempToken, username, displayName || undefined, oauthState);
      if (result.app_ticket) {
        returnToApp(result.app_ticket);
        return;
      }
      navigate('/', { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create account');
    } finally {
      setIsLoading(false);
    }
  };

  // Native flow: signed in — hand control back to the app
  if (appReturn) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
        <h1 className="text-2xl font-bold mb-4">Signed in!</h1>
        <p className="text-gray-400 mb-6">Returning to the KnightBall app…</p>
        <a
          href={appReturn}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded font-medium"
        >
          Open KnightBall
        </a>
      </div>
    );
  }

  // Username picker for new Google users
  if (needsUsername) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
        <h1 className="text-2xl font-bold mb-6">KnightBall</h1>
        <div className="bg-gray-800 rounded-lg p-6 max-w-sm w-full">
          <h2 className="text-xl font-bold mb-2">Choose a Username</h2>
          <p className="text-sm text-gray-400 mb-4">
            Signed in as {needsUsername.email}. Pick a username to complete your account.
          </p>

          <form onSubmit={handleCompleteSignup} className="space-y-4">
            {error && (
              <div className="bg-red-600 text-white px-3 py-2 rounded text-sm">{error}</div>
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
              <p className="text-xs text-gray-500 mt-1">3-32 characters, letters, numbers, underscores</p>
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

            <button
              type="submit"
              disabled={isLoading}
              className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
            >
              {isLoading ? 'Creating...' : 'Create Account'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <h1 className="text-2xl font-bold mb-6">KnightBall</h1>
      {isLoading && (
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p>Signing in with Google...</p>
        </div>
      )}
      {error && (
        <div className="text-center max-w-sm">
          <div className="bg-red-600 px-4 py-2 rounded mb-4">{error}</div>
          <a href="/" className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors inline-block">
            Back to KnightBall
          </a>
        </div>
      )}
    </div>
  );
}

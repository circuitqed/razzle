/**
 * Sign in with Apple — native app only (App Store guideline 4.8 requires it
 * alongside any third-party login). The sheet is fully native, so unlike the
 * Google flow there's no browser round-trip: the plugin returns an identity
 * token which the server verifies against Apple's JWKS.
 *
 * New users pick a username inline (Apple has no web page to host the picker,
 * unlike the Google Safari flow).
 */
import { useState } from 'react';
import { SignInWithApple } from '@capacitor-community/apple-sign-in';
import { useAuth } from '../contexts/AuthContext';
import { isNativeApp } from '../api/base';

interface AppleSignInButtonProps {
  onError?: (msg: string) => void;
}

export default function AppleSignInButton({ onError }: AppleSignInButtonProps) {
  const { appleAuth, appleComplete } = useAuth();
  const [needsUsername, setNeedsUsername] = useState<{ tempToken: string; suggestedName: string } | null>(null);
  const [username, setUsername] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isNativeApp) return null;

  const handleClick = async () => {
    setError(null);
    try {
      const result = await SignInWithApple.authorize({
        clientId: 'com.lazybrains.knightball',
        redirectURI: 'https://knightball.org/auth/apple/callback',
        scopes: 'email name',
      });
      const identityToken = result.response?.identityToken;
      if (!identityToken) {
        onError?.('Apple sign-in did not return a token');
        return;
      }
      // Apple only sends the name on FIRST authorization
      const name = [result.response?.givenName, result.response?.familyName]
        .filter(Boolean).join(' ') || null;
      const auth = await appleAuth(identityToken, name);
      if (auth.status === 'needs_username') {
        setNeedsUsername({ tempToken: auth.temp_token!, suggestedName: auth.suggested_name || '' });
      }
      // logged_in: AuthContext set the user; modals auto-close
    } catch (err: any) {
      // User cancelling the sheet is not an error worth surfacing
      const msg = String(err?.message ?? err);
      if (!/cancel/i.test(msg)) onError?.(`Apple sign-in failed: ${msg}`);
    }
  };

  const handleCompleteSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!needsUsername) return;
    setError(null);
    if (!/^[a-zA-Z0-9_]{3,32}$/.test(username)) {
      setError('Username: 3-32 characters, letters, numbers, underscores');
      return;
    }
    setIsLoading(true);
    try {
      await appleComplete(needsUsername.tempToken, username, needsUsername.suggestedName || undefined);
      setNeedsUsername(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create account');
    } finally {
      setIsLoading(false);
    }
  };

  if (needsUsername) {
    return (
      <form onSubmit={handleCompleteSignup} className="space-y-2">
        {error && <div className="bg-red-600 text-white px-3 py-2 rounded text-sm">{error}</div>}
        <label className="block text-sm text-gray-300">Choose a username to finish signing up</label>
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
        <button
          type="submit"
          disabled={isLoading}
          className="w-full px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
        >
          {isLoading ? 'Creating account…' : 'Continue'}
        </button>
      </form>
    );
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className="w-full flex items-center justify-center gap-3 px-4 py-2.5 bg-black hover:bg-gray-900 text-white rounded font-medium transition-colors border border-gray-600"
    >
      <svg width="16" height="18" viewBox="0 0 16 18" xmlns="http://www.w3.org/2000/svg" fill="currentColor">
        <path d="M13.06 9.56c-.02-2.02 1.65-2.99 1.72-3.04-.94-1.37-2.4-1.56-2.92-1.58-1.24-.13-2.42.73-3.05.73-.63 0-1.6-.71-2.63-.69-1.35.02-2.6.79-3.3 2-1.4 2.44-.36 6.05 1.01 8.03.67.97 1.47 2.06 2.52 2.02 1.01-.04 1.39-.65 2.61-.65 1.22 0 1.56.65 2.63.63 1.09-.02 1.78-.99 2.44-1.96.77-1.13 1.09-2.22 1.11-2.27-.02-.01-2.12-.81-2.14-3.22z"/>
        <path d="M11.06 3.62c.56-.68.93-1.62.83-2.56-.8.03-1.77.53-2.35 1.2-.51.6-.97 1.56-.85 2.48.9.07 1.81-.45 2.37-1.12z"/>
      </svg>
      Sign in with Apple
    </button>
  );
}

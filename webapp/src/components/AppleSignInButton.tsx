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
      // ASAuthorizationError 1001 = canceled — the user dismissed the sheet
      // (or iOS closed it). Not an error worth surfacing.
      const msg = String(err?.message ?? err);
      if (/cancel/i.test(msg) || /error 1001/i.test(msg)) return;
      // 1000 = unknown — most often the device isn't signed into iCloud
      if (/error 1000/i.test(msg)) {
        onError?.('Apple sign-in needs an iCloud account — check Settings › Apple ID.');
        return;
      }
      onError?.(`Apple sign-in failed: ${msg}`);
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
      <svg width="14" height="18" viewBox="0 0 384 512" xmlns="http://www.w3.org/2000/svg" fill="currentColor" aria-hidden="true">
        <path d="M318.7 268.7c-.2-36.7 16.4-64.4 50-84.8-18.8-26.9-47.2-41.7-84.7-44.6-35.5-2.8-74.3 20.7-88.5 20.7-15 0-49.4-19.7-76.4-19.7C63.3 141.2 4 184.8 4 273.5q0 39.3 14.4 81.2c12.8 36.7 59 126.7 107.2 125.2 25.2-.6 43-17.9 75.8-17.9 31.8 0 48.3 17.9 76.4 17.9 48.6-.7 90.4-82.5 102.6-119.3-65.2-30.7-61.7-90-61.7-91.9zm-56.6-164.2c27.3-32.4 24.8-61.9 24-72.5-24.1 1.4-52 16.4-67.9 34.9-17.5 19.8-27.8 44.3-25.6 71.9 26.1 2 49.9-11.4 69.5-34.3z"/>
      </svg>
      Sign in with Apple
    </button>
  );
}

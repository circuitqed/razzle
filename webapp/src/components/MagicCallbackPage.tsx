import { useState, useEffect, useRef } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

/**
 * Handles the passwordless sign-in link (/auth/magic?token=…).
 *
 * The email link is a Universal Link: on iOS with the app installed it opens
 * here inside the app (routed by NativeLinkHandler), so the session token is
 * captured natively; otherwise it opens the same page on the web. The token
 * is single-use, so whichever surface opens it wins.
 */
export default function MagicCallbackPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { verifyMagicLink } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const ranRef = useRef(false);

  const token = searchParams.get('token');

  useEffect(() => {
    if (ranRef.current) return; // token is single-use; never verify twice
    ranRef.current = true;
    if (!token) {
      setError('This sign-in link is missing its token.');
      return;
    }
    verifyMagicLink(token)
      .then(() => navigate('/', { replace: true }))
      .catch((err) => setError(err instanceof Error ? err.message : 'Sign-in link is invalid or expired.'));
  }, [token, verifyMagicLink, navigate]);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <h1 className="text-2xl font-bold mb-4">KnightBall</h1>
      {error ? (
        <>
          <p className="text-red-300 mb-6">{error}</p>
          <button
            onClick={() => navigate('/', { replace: true })}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded font-medium"
          >
            Back to KnightBall
          </button>
        </>
      ) : (
        <p className="text-gray-400">Signing you in…</p>
      )}
    </div>
  );
}

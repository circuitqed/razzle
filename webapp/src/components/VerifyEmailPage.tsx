import { useState, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { verifyEmail } from '../api/auth';

export default function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (!token) {
      setStatus('error');
      setMessage('Missing verification token.');
      return;
    }

    verifyEmail(token)
      .then(() => {
        setStatus('success');
        setMessage('Your email has been verified!');
      })
      .catch((err) => {
        setStatus('error');
        setMessage(err instanceof Error ? err.message : 'Verification failed. The link may be expired or already used.');
      });
  }, [token]);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <h1 className="text-2xl font-bold mb-6">KnightBall</h1>

      {status === 'loading' && (
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p>Verifying your email...</p>
        </div>
      )}

      {status === 'success' && (
        <div className="text-center max-w-sm">
          <div className="text-4xl mb-4">{'\u2705'}</div>
          <p className="text-green-400 text-lg mb-4">{message}</p>
          <Link
            to="/"
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors inline-block"
          >
            Go to KnightBall
          </Link>
        </div>
      )}

      {status === 'error' && (
        <div className="text-center max-w-sm">
          <div className="text-4xl mb-4">{'\u274C'}</div>
          <p className="text-red-400 text-lg mb-4">{message}</p>
          <Link
            to="/"
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors inline-block"
          >
            Go to KnightBall
          </Link>
        </div>
      )}
    </div>
  );
}

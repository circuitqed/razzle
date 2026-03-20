import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { resendVerification } from '../api/auth';

export default function EmailVerificationBanner() {
  const { user } = useAuth();
  const [dismissed, setDismissed] = useState(false);
  const [resending, setResending] = useState(false);
  const [resent, setResent] = useState(false);

  // Only show for logged-in users with unverified email
  if (!user || !user.email || user.email_verified || dismissed) {
    return null;
  }

  const handleResend = async () => {
    setResending(true);
    try {
      await resendVerification();
      setResent(true);
    } catch {
      // Silently fail - the user can try again
    } finally {
      setResending(false);
    }
  };

  return (
    <div className="bg-yellow-900/50 border-b border-yellow-700 px-4 py-2 flex items-center justify-center gap-3 text-sm">
      <span className="text-yellow-200">
        {resent
          ? 'Verification email sent! Check your inbox.'
          : 'Please verify your email address.'}
      </span>
      {!resent && (
        <button
          onClick={handleResend}
          disabled={resending}
          className="text-yellow-400 hover:text-yellow-300 underline disabled:opacity-50"
        >
          {resending ? 'Sending...' : 'Resend'}
        </button>
      )}
      <button
        onClick={() => setDismissed(true)}
        className="text-yellow-500 hover:text-yellow-300 ml-2"
        title="Dismiss"
      >
        {'\u2715'}
      </button>
    </div>
  );
}

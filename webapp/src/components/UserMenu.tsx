import { useState, useRef, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface UserMenuProps {
  onOpenLogin: () => void;
  onOpenRegister: () => void;
  onOpenBrowser: () => void;
}

export default function UserMenu({ onOpenLogin, onOpenRegister, onOpenBrowser }: UserMenuProps) {
  const { user, isAuthenticated, logout, deleteAccount, isLoading } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Reset the delete confirmation whenever the menu closes
  useEffect(() => {
    if (!isOpen) setConfirmingDelete(false);
  }, [isOpen]);

  const handleLogout = async () => {
    await logout();
    setIsOpen(false);
  };

  const handleDeleteAccount = async () => {
    setIsDeleting(true);
    try {
      await deleteAccount();
      setIsOpen(false);
      setConfirmingDelete(false);
    } catch (err) {
      console.error('Account deletion failed:', err);
    } finally {
      setIsDeleting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="px-3 py-1 text-gray-500 text-sm">
        Loading...
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex gap-2">
        <button
          onClick={onOpenLogin}
          className="px-3 py-1 text-sm text-gray-300 hover:text-white transition-colors"
        >
          Login
        </button>
        <button
          onClick={onOpenRegister}
          className="px-3 py-1 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
        >
          Sign Up
        </button>
      </div>
    );
  }

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1 rounded hover:bg-gray-700 transition-colors"
      >
        <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-medium">
          {(user?.username || user?.display_name || '?')[0].toUpperCase()}
        </div>
        <span className="text-sm text-gray-200 hidden sm:inline">
          {user?.username || user?.display_name}
        </span>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-56 bg-gray-800 rounded-lg shadow-lg border border-gray-700 py-1 z-50">
          <div className="px-4 py-2 border-b border-gray-700">
            <p className="text-sm font-medium text-white">{user?.display_name || user?.username}</p>
            <p className="text-xs text-gray-400">@{user?.username}</p>
            {user?.email && (
              <p className="text-xs text-gray-500 mt-1 truncate">
                {user.email}
                {user.email_verified ? (
                  <span className="text-green-400 ml-1" title="Verified">{'\u2713'}</span>
                ) : (
                  <span className="text-yellow-400 ml-1" title="Unverified">{'\u26A0'}</span>
                )}
              </p>
            )}
            {user?.auth_provider === 'google' && (
              <p className="text-xs text-gray-500 mt-0.5">Signed in with Google</p>
            )}
          </div>

          <button
            onClick={() => {
              setIsOpen(false);
              onOpenBrowser();
            }}
            className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-gray-700 transition-colors"
          >
            My Games
          </button>

          <button
            onClick={handleLogout}
            className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-gray-700 transition-colors"
          >
            Logout
          </button>

          {/* Account deletion — required in-app by App Store guideline 5.1.1(v) */}
          {confirmingDelete ? (
            <div className="px-4 py-3 border-t border-gray-700 space-y-2">
              <p className="text-xs text-red-300">
                Permanently delete your account and sign-in details? Your finished
                games remain anonymized. This cannot be undone.
              </p>
              <div className="flex gap-2">
                <button
                  onClick={handleDeleteAccount}
                  disabled={isDeleting}
                  className="flex-1 px-3 py-1.5 text-sm bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white rounded transition-colors"
                >
                  {isDeleting ? 'Deleting…' : 'Delete forever'}
                </button>
                <button
                  onClick={() => setConfirmingDelete(false)}
                  disabled={isDeleting}
                  className="flex-1 px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setConfirmingDelete(true)}
              className="w-full px-4 py-2 text-left text-xs text-gray-500 hover:bg-gray-700 hover:text-red-400 transition-colors border-t border-gray-700"
            >
              Delete Account…
            </button>
          )}
        </div>
      )}
    </div>
  );
}

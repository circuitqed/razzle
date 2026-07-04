/**
 * Auth modals must close when sign-in completes OUTSIDE their own form —
 * the native Google flow finishes via a knightball:// deep link (Safari
 * round-trip), so isAuthenticated flips while the modal is still open.
 */
import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import LoginModal from '../components/LoginModal';
import RegisterModal from '../components/RegisterModal';

const mockAuth = {
  user: null as unknown,
  isLoading: false,
  isAuthenticated: false,
  login: vi.fn(),
  register: vi.fn(),
  loginWithEmail: vi.fn(),
  registerWithEmail: vi.fn(),
  googleAuth: vi.fn(),
  googleComplete: vi.fn(),
  logout: vi.fn(),
  refreshUser: vi.fn(),
};

vi.mock('../contexts/AuthContext', () => ({
  useAuth: () => mockAuth,
}));

vi.mock('../components/GoogleSignInButton', () => ({
  default: () => null,
}));

describe('auth modals close on out-of-band sign-in', () => {
  it('LoginModal calls onClose when authenticated while open', () => {
    const onClose = vi.fn();
    mockAuth.isAuthenticated = false;
    const { rerender } = render(
      <LoginModal isOpen={true} onClose={onClose} onSwitchToRegister={vi.fn()} onForgotPassword={vi.fn()} />
    );
    expect(onClose).not.toHaveBeenCalled();

    mockAuth.isAuthenticated = true;
    rerender(
      <LoginModal isOpen={true} onClose={onClose} onSwitchToRegister={vi.fn()} onForgotPassword={vi.fn()} />
    );
    expect(onClose).toHaveBeenCalled();
  });

  it('RegisterModal calls onClose when authenticated while open', () => {
    const onClose = vi.fn();
    mockAuth.isAuthenticated = false;
    const { rerender } = render(
      <RegisterModal isOpen={true} onClose={onClose} onSwitchToLogin={vi.fn()} />
    );
    expect(onClose).not.toHaveBeenCalled();

    mockAuth.isAuthenticated = true;
    rerender(
      <RegisterModal isOpen={true} onClose={onClose} onSwitchToLogin={vi.fn()} />
    );
    expect(onClose).toHaveBeenCalled();
  });
});

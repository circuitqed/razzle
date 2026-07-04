import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import * as authApi from '../api/auth';
import { App as CapacitorApp } from '@capacitor/app';
import { Browser } from '@capacitor/browser';
import { isNativeApp, takePkceVerifier } from '../api/base';
import type { User, GoogleAuthResponse, AuthResponse, AppleAuthResponse } from '../api/auth';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, displayName?: string) => Promise<void>;
  loginWithEmail: (email: string, password: string) => Promise<void>;
  registerWithEmail: (email: string, username: string, password: string, displayName?: string) => Promise<void>;
  googleAuth: (credential: string, state?: string | null) => Promise<GoogleAuthResponse>;
  googleComplete: (tempToken: string, username: string, displayName?: string, state?: string | null) => Promise<AuthResponse>;
  appleAuth: (identityToken: string, displayName?: string | null) => Promise<AppleAuthResponse>;
  appleComplete: (tempToken: string, username: string, displayName?: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    try {
      const currentUser = await authApi.getCurrentUser();
      setUser(currentUser);
    } catch (error) {
      setUser(null);
    }
  }, []);

  // Check auth state on mount
  useEffect(() => {
    const checkAuth = async () => {
      setIsLoading(true);
      try {
        await refreshUser();
      } finally {
        setIsLoading(false);
      }
    };
    checkAuth();
  }, [refreshUser]);

  const login = useCallback(async (username: string, password: string) => {
    const response = await authApi.login(username, password);
    setUser(response.user);
  }, []);

  const register = useCallback(async (username: string, password: string, displayName?: string) => {
    const response = await authApi.register(username, password, displayName);
    setUser(response.user);
  }, []);

  const loginWithEmail = useCallback(async (email: string, password: string) => {
    const response = await authApi.loginWithEmail(email, password);
    setUser(response.user);
  }, []);

  const registerWithEmail = useCallback(async (email: string, username: string, password: string, displayName?: string) => {
    const response = await authApi.registerWithEmail(email, username, password, displayName);
    setUser(response.user);
  }, []);

  const googleAuth = useCallback(async (credential: string, state?: string | null): Promise<GoogleAuthResponse> => {
    const response = await authApi.googleAuth(credential, state);
    if (response.status === 'logged_in' && response.user) {
      setUser(response.user);
    }
    return response;
  }, []);

  const googleComplete = useCallback(async (tempToken: string, username: string, displayName?: string, state?: string | null) => {
    const response = await authApi.googleComplete(tempToken, username, displayName, state);
    setUser(response.user);
    return response;
  }, []);

  const appleAuth = useCallback(async (identityToken: string, displayName?: string | null): Promise<AppleAuthResponse> => {
    const response = await authApi.appleAuth(identityToken, displayName);
    if (response.status === 'logged_in' && response.user) {
      setUser(response.user);
    }
    return response;
  }, []);

  const appleComplete = useCallback(async (tempToken: string, username: string, displayName?: string) => {
    const response = await authApi.appleComplete(tempToken, username, displayName);
    setUser(response.user);
  }, []);

  const logout = useCallback(async () => {
    await authApi.logout();
    setUser(null);
  }, []);

  // Native OAuth return: the system-browser flow deep-links back with
  // knightball://auth?ticket=<one-time>; exchange it for a session.
  useEffect(() => {
    if (!isNativeApp) return;
    const sub = CapacitorApp.addListener('appUrlOpen', async ({ url }) => {
      try {
        const parsed = new URL(url);
        if (parsed.protocol !== 'knightball:' || parsed.hostname !== 'auth') return;
        const ticket = parsed.searchParams.get('ticket');
        if (!ticket) return;
        Browser.close().catch(() => { /* browser may already be closed */ });
        // PKCE: only exchange tickets for a flow THIS app started. A deep
        // link arriving with no pending verifier is unsolicited — ignore it.
        const verifier = takePkceVerifier();
        if (!verifier) {
          console.warn('[auth] ignoring unsolicited auth deep link');
          return;
        }
        const { user: signedIn } = await authApi.exchangeAppTicket(ticket, verifier);
        setUser(signedIn);
      } catch (err) {
        console.error('[auth] app ticket exchange failed:', err);
      }
    });
    return () => { sub.then((h) => h.remove()); };
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        loginWithEmail,
        registerWithEmail,
        googleAuth,
        googleComplete,
        appleAuth,
        appleComplete,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import * as authApi from '../api/auth';
import type { User } from '../api/auth';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, displayName?: string) => Promise<void>;
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

  const logout = useCallback(async () => {
    await authApi.logout();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
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

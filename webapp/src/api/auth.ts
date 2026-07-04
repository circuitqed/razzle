/**
 * Authentication API client
 */

import { API_BASE } from './base';

export interface User {
  user_id: string;
  username: string;
  display_name: string | null;
  created_at: string;
  last_login_at: string | null;
  email: string | null;
  email_verified: boolean;
  auth_provider: 'local' | 'google';
}

export interface AuthResponse {
  user: User;
  message: string;
}

export interface GoogleAuthResponse {
  status: 'logged_in' | 'needs_username';
  user: User | null;
  temp_token: string | null;
  email: string | null;
  suggested_name: string | null;
}

class AuthAPIError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
    this.name = 'AuthAPIError';
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: 'include', // Include cookies
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }));
    throw new AuthAPIError(response.status, error.code || 'UNKNOWN', error.detail || error.message);
  }

  return response.json();
}

// Legacy username-only registration
export async function register(
  username: string,
  password: string,
  displayName?: string
): Promise<AuthResponse> {
  return request('/auth/register', {
    method: 'POST',
    body: JSON.stringify({
      username,
      password,
      display_name: displayName,
    }),
  });
}

// Legacy username-only login
export async function login(username: string, password: string): Promise<AuthResponse> {
  return request('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
}

// Email registration
export async function registerWithEmail(
  email: string,
  username: string,
  password: string,
  displayName?: string
): Promise<AuthResponse> {
  return request('/auth/register/email', {
    method: 'POST',
    body: JSON.stringify({
      email,
      username,
      password,
      display_name: displayName,
    }),
  });
}

// Email/username login
export async function loginWithEmail(email: string, password: string): Promise<AuthResponse> {
  return request('/auth/login/email', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
}

// Google OAuth
export async function googleAuth(credential: string): Promise<GoogleAuthResponse> {
  // Include redirect_uri so backend can exchange auth codes
  const redirectUri = `${window.location.origin}/auth/google/callback`;
  return request('/auth/google', {
    method: 'POST',
    body: JSON.stringify({ credential, redirect_uri: redirectUri }),
  });
}

// Complete Google sign-up with username
export async function googleComplete(
  tempToken: string,
  username: string,
  displayName?: string
): Promise<AuthResponse> {
  return request('/auth/google/complete', {
    method: 'POST',
    body: JSON.stringify({
      temp_token: tempToken,
      username,
      display_name: displayName,
    }),
  });
}

// Email verification
export async function verifyEmail(token: string): Promise<{ message: string }> {
  return request('/auth/verify-email', {
    method: 'POST',
    body: JSON.stringify({ token }),
  });
}

// Resend verification email
export async function resendVerification(): Promise<{ message: string }> {
  return request('/auth/resend-verification', { method: 'POST' });
}

// Forgot password
export async function forgotPassword(email: string): Promise<{ message: string }> {
  return request('/auth/forgot-password', {
    method: 'POST',
    body: JSON.stringify({ email }),
  });
}

// Reset password
export async function resetPassword(token: string, password: string): Promise<{ message: string }> {
  return request('/auth/reset-password', {
    method: 'POST',
    body: JSON.stringify({ token, password }),
  });
}

export async function logout(): Promise<{ message: string }> {
  return request('/auth/logout', { method: 'POST' });
}

export async function getCurrentUser(): Promise<User | null> {
  try {
    return await request<User>('/auth/me');
  } catch (error) {
    if (error instanceof AuthAPIError && error.status === 401) {
      return null;
    }
    throw error;
  }
}

export { AuthAPIError };

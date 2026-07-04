/**
 * Authentication API client
 */

import { API_BASE, setNativeAuthToken } from './base';

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
  /** Session JWT — returned to native clients only. */
  token?: string | null;
  /** One-time app-return ticket for native OAuth flows. */
  app_ticket?: string | null;
}

export interface GoogleAuthResponse {
  status: 'logged_in' | 'needs_username';
  user: User | null;
  temp_token: string | null;
  email: string | null;
  suggested_name: string | null;
  /** One-time app-return ticket for native OAuth flows. */
  app_ticket?: string | null;
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

  const data = await response.json();
  // Native app: auth endpoints return the session JWT in the body (the
  // HttpOnly cookie can't persist in the webview) — store it for the
  // Authorization header.
  if (data && typeof data === 'object' && typeof data.token === 'string') {
    setNativeAuthToken(data.token);
  }
  return data as T;
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
export async function googleAuth(credential: string, state?: string | null): Promise<GoogleAuthResponse> {
  // Include redirect_uri so backend can exchange auth codes
  const redirectUri = `${window.location.origin}/auth/google/callback`;
  return request('/auth/google', {
    method: 'POST',
    body: JSON.stringify({ credential, redirect_uri: redirectUri, state }),
  });
}

// Complete Google sign-up with username
export async function googleComplete(
  tempToken: string,
  username: string,
  displayName?: string,
  state?: string | null
): Promise<AuthResponse> {
  return request('/auth/google/complete', {
    method: 'POST',
    body: JSON.stringify({
      temp_token: tempToken,
      username,
      display_name: displayName,
      state,
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

export interface AppleAuthResponse {
  status: 'logged_in' | 'needs_username';
  user: User | null;
  temp_token: string | null;
  email: string | null;
  suggested_name: string | null;
  token?: string | null;
}

/** Sign in with Apple: verify the native sheet's identity token (native only). */
export async function appleAuth(identityToken: string, displayName?: string | null): Promise<AppleAuthResponse> {
  return request('/auth/apple', {
    method: 'POST',
    body: JSON.stringify({ identity_token: identityToken, display_name: displayName }),
  });
}

/** Complete Apple sign-up with a chosen username. */
export async function appleComplete(
  tempToken: string,
  username: string,
  displayName?: string
): Promise<AuthResponse> {
  return request('/auth/apple/complete', {
    method: 'POST',
    body: JSON.stringify({ temp_token: tempToken, username, display_name: displayName }),
  });
}

/** Exchange a one-time native OAuth app ticket for a session (native only). */
export async function exchangeAppTicket(ticket: string, codeVerifier: string): Promise<{ token: string; user: User }> {
  const result = await request<{ token: string; user: User }>('/auth/app-ticket/exchange', {
    method: 'POST',
    body: JSON.stringify({ ticket, code_verifier: codeVerifier }),
  });
  setNativeAuthToken(result.token);
  return result;
}

/** Permanently delete the authenticated account (guideline 5.1.1(v)). */
export async function deleteAccount(): Promise<{ message: string }> {
  const result = await request<{ message: string }>('/auth/account', { method: 'DELETE' });
  setNativeAuthToken(null);
  return result;
}

export async function logout(): Promise<{ message: string }> {
  setNativeAuthToken(null);
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

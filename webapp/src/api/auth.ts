/**
 * Authentication API client
 */

const API_BASE = '/api';

export interface User {
  user_id: string;
  username: string;
  display_name: string | null;
  created_at: string;
  last_login_at: string | null;
}

export interface AuthResponse {
  user: User;
  message: string;
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

export async function login(username: string, password: string): Promise<AuthResponse> {
  return request('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
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

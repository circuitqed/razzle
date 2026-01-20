import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { AuthProvider, useAuth } from '../contexts/AuthContext'
import * as authApi from '../api/auth'

// Mock the auth API
vi.mock('../api/auth', () => ({
  register: vi.fn(),
  login: vi.fn(),
  logout: vi.fn(),
  getCurrentUser: vi.fn(),
}))

// Test component that uses the auth hook
function TestAuthComponent() {
  const { user, isAuthenticated, isLoading, login, register, logout } = useAuth()

  return (
    <div>
      <div data-testid="loading">{isLoading ? 'loading' : 'loaded'}</div>
      <div data-testid="authenticated">{isAuthenticated ? 'yes' : 'no'}</div>
      <div data-testid="username">{user?.username || 'none'}</div>
      <button onClick={() => login('testuser', 'password')}>Login</button>
      <button onClick={() => register('newuser', 'password', 'New User')}>Register</button>
      <button onClick={() => logout()}>Logout</button>
    </div>
  )
}

describe('AuthContext', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Default: no user logged in
    vi.mocked(authApi.getCurrentUser).mockResolvedValue(null)
  })

  it('shows not authenticated initially when no user', async () => {
    render(
      <AuthProvider>
        <TestAuthComponent />
      </AuthProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded')
    })

    expect(screen.getByTestId('authenticated')).toHaveTextContent('no')
    expect(screen.getByTestId('username')).toHaveTextContent('none')
  })

  it('shows authenticated when user exists', async () => {
    vi.mocked(authApi.getCurrentUser).mockResolvedValue({
      user_id: 'user123',
      username: 'existinguser',
      display_name: 'Existing User',
      created_at: '2024-01-01',
      last_login_at: '2024-01-01',
    })

    render(
      <AuthProvider>
        <TestAuthComponent />
      </AuthProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded')
    })

    expect(screen.getByTestId('authenticated')).toHaveTextContent('yes')
    expect(screen.getByTestId('username')).toHaveTextContent('existinguser')
  })

  it('handles login correctly', async () => {
    const mockUser = {
      user_id: 'user123',
      username: 'testuser',
      display_name: 'Test User',
      created_at: '2024-01-01',
      last_login_at: '2024-01-01',
    }

    vi.mocked(authApi.login).mockResolvedValue({
      user: mockUser,
      message: 'Login successful',
    })

    render(
      <AuthProvider>
        <TestAuthComponent />
      </AuthProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded')
    })

    fireEvent.click(screen.getByText('Login'))

    await waitFor(() => {
      expect(screen.getByTestId('authenticated')).toHaveTextContent('yes')
    })

    expect(screen.getByTestId('username')).toHaveTextContent('testuser')
    expect(authApi.login).toHaveBeenCalledWith('testuser', 'password')
  })

  it('handles registration correctly', async () => {
    const mockUser = {
      user_id: 'user456',
      username: 'newuser',
      display_name: 'New User',
      created_at: '2024-01-01',
      last_login_at: null,
    }

    vi.mocked(authApi.register).mockResolvedValue({
      user: mockUser,
      message: 'Account created',
    })

    render(
      <AuthProvider>
        <TestAuthComponent />
      </AuthProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded')
    })

    fireEvent.click(screen.getByText('Register'))

    await waitFor(() => {
      expect(screen.getByTestId('authenticated')).toHaveTextContent('yes')
    })

    expect(screen.getByTestId('username')).toHaveTextContent('newuser')
    expect(authApi.register).toHaveBeenCalledWith('newuser', 'password', 'New User')
  })

  it('handles logout correctly', async () => {
    // Start logged in
    vi.mocked(authApi.getCurrentUser).mockResolvedValue({
      user_id: 'user123',
      username: 'loggeduser',
      display_name: 'Logged User',
      created_at: '2024-01-01',
      last_login_at: '2024-01-01',
    })

    vi.mocked(authApi.logout).mockResolvedValue({ message: 'Logged out' })

    render(
      <AuthProvider>
        <TestAuthComponent />
      </AuthProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('authenticated')).toHaveTextContent('yes')
    })

    fireEvent.click(screen.getByText('Logout'))

    await waitFor(() => {
      expect(screen.getByTestId('authenticated')).toHaveTextContent('no')
    })

    expect(authApi.logout).toHaveBeenCalled()
  })
})

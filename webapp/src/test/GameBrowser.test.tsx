import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import GameBrowser from '../components/GameBrowser'
import { AuthProvider } from '../contexts/AuthContext'
import * as gamesApi from '../api/games'
import * as authApi from '../api/auth'

// Mock the APIs
vi.mock('../api/games', () => ({
  listGames: vi.fn(),
  getGameFull: vi.fn(),
  analyzePosition: vi.fn(),
  analyzeGame: vi.fn(),
}))

vi.mock('../api/auth', () => ({
  getCurrentUser: vi.fn(),
  login: vi.fn(),
  logout: vi.fn(),
  register: vi.fn(),
}))

const mockGames = {
  games: [
    {
      game_id: 'game1',
      player1_type: 'human',
      player2_type: 'ai',
      player1_user_id: null,
      player2_user_id: null,
      player1_username: null,
      player2_username: null,
      status: 'finished',
      winner: 0,
      move_count: 25,
      ply: 25,
      created_at: '2024-01-15T10:00:00Z',
      updated_at: '2024-01-15T10:30:00Z',
      ai_model_version: '/models/model_v1.pt',
    },
    {
      game_id: 'game2',
      player1_type: 'human',
      player2_type: 'human',
      player1_user_id: 'user1',
      player2_user_id: 'user2',
      player1_username: 'Alice',
      player2_username: 'Bob',
      status: 'playing',
      winner: null,
      move_count: 10,
      ply: 10,
      created_at: '2024-01-16T14:00:00Z',
      updated_at: '2024-01-16T14:15:00Z',
      ai_model_version: null,
    },
  ],
  total: 2,
  page: 1,
  per_page: 15,
  total_pages: 1,
}

function renderWithAuth(ui: React.ReactElement) {
  return render(<AuthProvider>{ui}</AuthProvider>)
}

describe('GameBrowser', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(authApi.getCurrentUser).mockResolvedValue(null)
    vi.mocked(gamesApi.listGames).mockResolvedValue(mockGames)
  })

  it('renders nothing when not open', () => {
    const onClose = vi.fn()
    const onSelectGame = vi.fn()

    renderWithAuth(
      <GameBrowser isOpen={false} onClose={onClose} onSelectGame={onSelectGame} />
    )

    expect(screen.queryByText('Game History')).not.toBeInTheDocument()
  })

  it('renders game list when open', async () => {
    const onClose = vi.fn()
    const onSelectGame = vi.fn()

    renderWithAuth(
      <GameBrowser isOpen={true} onClose={onClose} onSelectGame={onSelectGame} />
    )

    expect(screen.getByText('Game History')).toBeInTheDocument()

    await waitFor(() => {
      expect(gamesApi.listGames).toHaveBeenCalled()
    })
  })

  it('displays games in the list', async () => {
    const onClose = vi.fn()
    const onSelectGame = vi.fn()

    renderWithAuth(
      <GameBrowser isOpen={true} onClose={onClose} onSelectGame={onSelectGame} />
    )

    await waitFor(() => {
      // Shows model filename (not full path) for AI games
      expect(screen.getByText('Human vs model_v1.pt')).toBeInTheDocument()
    })

    // Shows usernames for human vs human games
    expect(screen.getByText('Alice vs Bob')).toBeInTheDocument()
    // Use getAllByText since these may appear in filter dropdowns too
    expect(screen.getAllByText('Blue Won').length).toBeGreaterThan(0)
    expect(screen.getAllByText('In Progress').length).toBeGreaterThan(0)
  })

  it('calls onSelectGame when replay button is clicked', async () => {
    const onClose = vi.fn()
    const onSelectGame = vi.fn()

    renderWithAuth(
      <GameBrowser isOpen={true} onClose={onClose} onSelectGame={onSelectGame} />
    )

    await waitFor(() => {
      expect(screen.getAllByText('Replay').length).toBeGreaterThan(0)
    })

    const replayButtons = screen.getAllByText('Replay')
    fireEvent.click(replayButtons[0])

    expect(onSelectGame).toHaveBeenCalledWith('game1')
  })

  it('calls onClose when close button is clicked', async () => {
    const onClose = vi.fn()
    const onSelectGame = vi.fn()

    renderWithAuth(
      <GameBrowser isOpen={true} onClose={onClose} onSelectGame={onSelectGame} />
    )

    await waitFor(() => {
      expect(screen.getByText('Game History')).toBeInTheDocument()
    })

    // Find the close button by its SVG content (the X icon)
    const closeButtons = screen.getAllByRole('button')
    // The close button is the one without text content
    const closeButton = closeButtons.find(btn => btn.querySelector('svg path'))
    if (closeButton) {
      fireEvent.click(closeButton)
      expect(onClose).toHaveBeenCalled()
    }
  })

  it('shows empty state when no games', async () => {
    vi.mocked(gamesApi.listGames).mockResolvedValue({
      games: [],
      total: 0,
      page: 1,
      per_page: 15,
      total_pages: 0,
    })

    const onClose = vi.fn()
    const onSelectGame = vi.fn()

    renderWithAuth(
      <GameBrowser isOpen={true} onClose={onClose} onSelectGame={onSelectGame} />
    )

    await waitFor(() => {
      expect(screen.getByText('No games found')).toBeInTheDocument()
    })
  })

  it('shows error when API fails', async () => {
    vi.mocked(gamesApi.listGames).mockRejectedValue(new Error('API Error'))

    const onClose = vi.fn()
    const onSelectGame = vi.fn()

    renderWithAuth(
      <GameBrowser isOpen={true} onClose={onClose} onSelectGame={onSelectGame} />
    )

    await waitFor(() => {
      expect(screen.getByText('API Error')).toBeInTheDocument()
    })
  })

  it('applies status filter', async () => {
    const onClose = vi.fn()
    const onSelectGame = vi.fn()

    renderWithAuth(
      <GameBrowser isOpen={true} onClose={onClose} onSelectGame={onSelectGame} />
    )

    await waitFor(() => {
      expect(gamesApi.listGames).toHaveBeenCalled()
    })

    // Find all select elements (comboboxes)
    const selects = screen.getAllByRole('combobox')
    // Status filter is the first select
    const statusSelect = selects[0]
    fireEvent.change(statusSelect, { target: { value: 'finished' } })

    await waitFor(() => {
      expect(gamesApi.listGames).toHaveBeenCalledWith(
        expect.objectContaining({ status: 'finished' })
      )
    })
  })
})

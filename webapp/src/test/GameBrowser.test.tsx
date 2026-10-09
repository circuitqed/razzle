import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import GameBrowser from '../components/GameBrowser'
import { AuthProvider } from '../contexts/AuthContext'
import * as gamesApi from '../api/games'
import * as authApi from '../api/auth'
import * as leaderboardApi from '../api/leaderboard'

// Mock the APIs
vi.mock('../api/games', () => ({
  listGames: vi.fn(),
  getGameFull: vi.fn(),
  analyzePosition: vi.fn(),
  analyzeGame: vi.fn(),
}))

vi.mock('../api/leaderboard', () => ({
  getPlayers: vi.fn(),
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
      ai_simulations: 800,
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
      ai_simulations: 0,
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

const ALICE = { player_id: 'human_u1', user_id: 'u1', username: 'Alice', display_name: 'Alice', elo_rating: 1200 }

function renderBrowser(onClose = vi.fn(), onSelectGame = vi.fn()) {
  renderWithAuth(<GameBrowser isOpen={true} onClose={onClose} onSelectGame={onSelectGame} />)
  return { onClose, onSelectGame }
}

/** Search for Alice and pick her from the dropdown. */
async function pickAlice() {
  await waitFor(() => expect(leaderboardApi.getPlayers).toHaveBeenCalled())
  fireEvent.change(screen.getByPlaceholderText('Search...'), { target: { value: 'ali' } })
  fireEvent.click(await screen.findByText('Alice', { selector: 'button span' }))
}

describe('GameBrowser', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(authApi.getCurrentUser).mockResolvedValue(null)
    vi.mocked(gamesApi.listGames).mockResolvedValue(mockGames)
    vi.mocked(leaderboardApi.getPlayers).mockResolvedValue([ALICE] as never)
  })

  it('renders nothing when not open', () => {
    renderWithAuth(<GameBrowser isOpen={false} onClose={vi.fn()} onSelectGame={vi.fn()} />)
    expect(screen.queryByText('Players’ games')).not.toBeInTheDocument()
  })

  it('has no global game list: asks for a player first', async () => {
    renderBrowser()
    expect(screen.getByText('Players’ games')).toBeInTheDocument()
    expect(await screen.findByText('Search for a player to see their games.')).toBeInTheDocument()
    expect(gamesApi.listGames).not.toHaveBeenCalled()
  })

  it("lists a player's games once picked", async () => {
    renderBrowser()
    await pickAlice()
    await waitFor(() => {
      expect(gamesApi.listGames).toHaveBeenCalledWith(expect.objectContaining({ player_id: 'u1' }))
    })
    expect(await screen.findByText('Human vs model_v1 · 800 sims')).toBeInTheDocument()
    expect(screen.getByText('Alice vs Bob')).toBeInTheDocument()
    expect(screen.getAllByText('Blue Won').length).toBeGreaterThan(0)
    expect(screen.getAllByText('In progress').length).toBeGreaterThan(0)
  })

  it('opens a game when its row is clicked', async () => {
    const { onSelectGame } = renderBrowser()
    await pickAlice()
    fireEvent.click(await screen.findByText('Human vs model_v1 · 800 sims'))
    expect(onSelectGame).toHaveBeenCalledWith('game1')
  })

  it('calls onClose when close button is clicked', async () => {
    const { onClose } = renderBrowser()
    const closeButton = screen.getAllByRole('button').find(btn => btn.querySelector('svg path'))
    expect(closeButton).toBeDefined()
    fireEvent.click(closeButton!)
    expect(onClose).toHaveBeenCalled()
  })

  it('shows empty state when the player has no games', async () => {
    vi.mocked(gamesApi.listGames).mockResolvedValue({ games: [], total: 0, page: 1, per_page: 15, total_pages: 0 })
    renderBrowser()
    await pickAlice()
    expect(await screen.findByText('No games found')).toBeInTheDocument()
  })

  it('shows error when API fails', async () => {
    vi.mocked(gamesApi.listGames).mockRejectedValue(new Error('API Error'))
    renderBrowser()
    await pickAlice()
    expect(await screen.findByText('API Error')).toBeInTheDocument()
  })

  it('applies status filter', async () => {
    renderBrowser()
    await pickAlice()
    await waitFor(() => expect(gamesApi.listGames).toHaveBeenCalled())
    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'abandoned' } })
    await waitFor(() => {
      expect(gamesApi.listGames).toHaveBeenCalledWith(expect.objectContaining({ status: 'abandoned', player_id: 'u1' }))
    })
  })
})

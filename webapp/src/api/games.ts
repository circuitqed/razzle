/**
 * Games listing and analysis API client
 */

const API_BASE = '/api';

export interface GameSummary {
  game_id: string;
  player1_type: string;
  player2_type: string;
  player1_user_id: string | null;
  player2_user_id: string | null;
  player1_username: string | null;
  player2_username: string | null;
  status: string;
  winner: number | null;
  move_count: number;
  ply: number;
  created_at: string;
  updated_at: string;
  ai_model_version: string | null;
}

export interface GameListResponse {
  games: GameSummary[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface GameFull {
  game_id: string;
  player1_type: string;
  player2_type: string;
  player1_user_id: string | null;
  player2_user_id: string | null;
  status: string;
  winner: number | null;
  ply: number;
  moves: number[];
  moves_algebraic: string[];
  created_at: string;
  updated_at: string;
  ai_model_version: string | null;
}

export interface MoveAnalysis {
  move: number;
  algebraic: string;
  visits: number;
  value: number;
  policy: number;
}

export interface AnalyzePositionResponse {
  value: number;
  legal_moves: number[];
  top_moves: MoveAnalysis[];
  time_ms: number;
}

export interface MoveClassification {
  move: number;
  algebraic: string;
  value_before: number;
  value_after: number;
  best_move: number;
  best_move_algebraic: string;
  best_value: number;
  delta: number;
  classification: 'best' | 'good' | 'inaccuracy' | 'mistake' | 'blunder';
}

export interface AnalyzeGameResponse {
  game_id: string;
  move_analyses: MoveClassification[];
  summary: {
    best: number;
    good: number;
    inaccuracy: number;
    mistake: number;
    blunder: number;
  };
}

class GamesAPIError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
    this.name = 'GamesAPIError';
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }));
    throw new GamesAPIError(response.status, error.code || 'UNKNOWN', error.detail || error.message);
  }

  return response.json();
}

export async function listGames(params?: {
  player_id?: string;
  status?: string;
  winner?: number;
  date_from?: string;
  date_to?: string;
  page?: number;
  per_page?: number;
}): Promise<GameListResponse> {
  const searchParams = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.set(key, String(value));
      }
    });
  }
  const queryString = searchParams.toString();
  return request(`/games${queryString ? `?${queryString}` : ''}`);
}

export async function getGameFull(gameId: string): Promise<GameFull> {
  return request(`/games/${gameId}/full`);
}

export async function analyzePosition(params: {
  pieces: [string, string];
  balls: [string, string];
  current_player: number;
  touched_mask?: string;
  has_passed?: boolean;
  last_knight_dst?: number;
  simulations?: number;
}): Promise<AnalyzePositionResponse> {
  return request('/analyze', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function analyzeGame(
  gameId: string,
  simulationsPerPosition?: number
): Promise<AnalyzeGameResponse> {
  const params = simulationsPerPosition
    ? `?simulations_per_position=${simulationsPerPosition}`
    : '';
  return request(`/games/${gameId}/analyze${params}`, { method: 'POST' });
}

export { GamesAPIError };

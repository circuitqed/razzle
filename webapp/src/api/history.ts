/**
 * Account game history + AI progress API client (signed-in users only).
 */

import { API_BASE } from './base';

export interface HistoryOpponent {
  type: 'ai' | 'human' | 'local';
  name: string;
  user_id?: string | null;
  username?: string | null;
  ai_level?: number | null;
  ai_model?: string | null;
  ai_simulations?: number | null;
}

export type HistoryResult = 'win' | 'loss' | 'draw' | 'in_progress' | 'abandoned' | 'aborted' | null;

export interface HistoryGame {
  game_id: string;
  mode: 'ai' | 'online' | 'local';
  game_mode: string;
  your_color: 0 | 1;
  opponent: HistoryOpponent;
  result: HistoryResult;
  winner: number | null;
  resigned: boolean;
  move_count: number;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
}

export interface HistoryResponse {
  games: HistoryGame[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface RecordCounts {
  wins: number;
  losses: number;
  draws: number;
  games: number;
}

export interface GameSummaryStats {
  vs_ai: RecordCounts & { by_level: (RecordCounts & { level: number })[] };
  vs_human: RecordCounts;
  highest_ai_level_beaten: number;
  auto_match_level: number | null;
  auto_match_level_updated_at: string | null;
}

export interface AIProgress {
  current_level: number | null;
  current_level_updated_at: string | null;
  highest_level_beaten: number;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

export function getMyGames(page = 1, perPage = 20): Promise<HistoryResponse> {
  return request(`/me/games?page=${page}&per_page=${perPage}`);
}

export function getMySummary(): Promise<GameSummaryStats> {
  return request('/me/summary');
}

export function getAIProgress(): Promise<AIProgress> {
  return request('/me/ai-progress');
}

export function putAIProgress(progress: Partial<AIProgress>): Promise<AIProgress> {
  return request('/me/ai-progress', { method: 'PUT', body: JSON.stringify(progress) });
}

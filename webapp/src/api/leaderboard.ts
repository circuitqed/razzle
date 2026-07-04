import type { PlayerProfile, PlayerProfileWithGames, PlayersListResponse, LeaderboardResponse } from '../types';

import { API_BASE } from './base';

/**
 * Get the ELO leaderboard.
 */
export async function getLeaderboard(
  playerType?: 'human' | 'ai',
  limit: number = 50,
  minGames: number = 1
): Promise<PlayerProfile[]> {
  const params = new URLSearchParams();
  if (playerType) params.set('player_type', playerType);
  params.set('limit', limit.toString());
  params.set('min_games', minGames.toString());

  const response = await fetch(`${API_BASE}/leaderboard?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch leaderboard: ${response.statusText}`);
  }
  const data: LeaderboardResponse = await response.json();
  return data.players;
}

/**
 * Get all players.
 */
export async function getPlayers(
  playerType?: 'human' | 'ai',
  limit: number = 100,
  minGames: number = 0
): Promise<PlayerProfile[]> {
  const params = new URLSearchParams();
  if (playerType) params.set('player_type', playerType);
  params.set('limit', limit.toString());
  params.set('min_games', minGames.toString());

  const response = await fetch(`${API_BASE}/players?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch players: ${response.statusText}`);
  }
  const data: PlayersListResponse = await response.json();
  return data.players;
}

/**
 * Get a player by ID with recent games.
 */
export async function getPlayer(
  playerId: string,
  includeGames: boolean = true,
  gamesLimit: number = 20
): Promise<PlayerProfileWithGames> {
  const params = new URLSearchParams();
  params.set('include_games', includeGames.toString());
  params.set('games_limit', gamesLimit.toString());

  const response = await fetch(`${API_BASE}/players/${encodeURIComponent(playerId)}?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch player: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Format player display for leaderboard.
 */
export function formatPlayerDisplay(player: PlayerProfile): string {
  if (player.player_type === 'ai') {
    const modelPart = player.model_version || 'unknown';
    const simsPart = player.simulations ? ` (${player.simulations} sims)` : '';
    return `${modelPart}${simsPart}`;
  }
  return player.username || player.display_name;
}

/**
 * Get ELO delta display string.
 */
export function getEloDelta(current: number, baseline: number = 1000): string {
  const delta = current - baseline;
  if (delta > 0) return `+${delta.toFixed(0)}`;
  if (delta < 0) return delta.toFixed(0);
  return '0';
}

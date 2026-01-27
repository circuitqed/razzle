import type { ArenaMatch, ArenaRating, ArenaMatchesResponse, ArenaRatingsResponse } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

/**
 * Get arena match history.
 */
export async function getArenaMatches(
  model?: string,
  simulations?: number,
  limit: number = 100
): Promise<ArenaMatch[]> {
  const params = new URLSearchParams();
  if (model) params.set('model', model);
  if (simulations != null) params.set('simulations', simulations.toString());
  params.set('limit', limit.toString());

  const response = await fetch(`${API_BASE}/arena/matches?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch arena matches: ${response.statusText}`);
  }
  const data: ArenaMatchesResponse = await response.json();
  return data.matches;
}

/**
 * Get arena ELO ratings.
 */
export async function getArenaRatings(
  simulations?: number,
  limit: number = 100
): Promise<ArenaRating[]> {
  const params = new URLSearchParams();
  if (simulations != null) params.set('simulations', simulations.toString());
  params.set('limit', limit.toString());

  const response = await fetch(`${API_BASE}/arena/ratings?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch arena ratings: ${response.statusText}`);
  }
  const data: ArenaRatingsResponse = await response.json();
  return data.ratings;
}

/**
 * Submit an arena match result.
 */
export async function submitArenaMatch(match: Omit<ArenaMatch, 'id' | 'created_at'>): Promise<{ id: number; status: string }> {
  const response = await fetch(`${API_BASE}/arena/matches`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(match),
  });
  if (!response.ok) {
    throw new Error(`Failed to submit arena match: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Compute ELO ratings from match history.
 */
export async function computeArenaRatings(
  anchorModel: string = 'initial',
  simulations?: number
): Promise<{ status: string; simulation_counts: number[]; ratings: Record<number, Record<string, { elo: number; games: number }>> }> {
  const response = await fetch(`${API_BASE}/arena/compute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ anchor_model: anchorModel, simulations }),
  });
  if (!response.ok) {
    throw new Error(`Failed to compute arena ratings: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Clear all arena data.
 */
export async function clearArenaData(): Promise<{ matches_deleted: number; ratings_deleted: number }> {
  const response = await fetch(`${API_BASE}/arena/clear`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(`Failed to clear arena data: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get unique simulation counts from ratings.
 */
export function getUniqueSimulationCounts(ratings: ArenaRating[]): number[] {
  const counts = new Set(ratings.map(r => r.simulations));
  return Array.from(counts).sort((a, b) => a - b);
}

/**
 * Filter ratings by simulation count.
 */
export function filterRatingsBySimulations(ratings: ArenaRating[], simulations: number): ArenaRating[] {
  return ratings.filter(r => r.simulations === simulations);
}

/**
 * Sort ratings by ELO (descending).
 */
export function sortRatingsByElo(ratings: ArenaRating[]): ArenaRating[] {
  return [...ratings].sort((a, b) => b.elo_rating - a.elo_rating);
}

/**
 * Sort ratings by iteration.
 */
export function sortRatingsByIteration(ratings: ArenaRating[]): ArenaRating[] {
  return [...ratings].sort((a, b) => a.iteration - b.iteration);
}

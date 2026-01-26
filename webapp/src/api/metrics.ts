import type { TrainingMetrics, TrainingMetricsResponse, LatestMetricsResponse, TrainingDashboardData } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

class MetricsAPIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'MetricsAPIError';
  }
}

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }));
    throw new MetricsAPIError(response.status, error.message || error.detail || 'Request failed');
  }

  return response.json();
}

/**
 * Get training metrics history.
 *
 * @param limit Maximum number of records to return (default: 100)
 * @param offset Number of records to skip (default: 0)
 * @returns List of metrics and total count
 */
export async function getTrainingMetrics(
  limit: number = 100,
  offset: number = 0
): Promise<TrainingMetricsResponse> {
  return request(`/training/metrics?limit=${limit}&offset=${offset}`);
}

/**
 * Get the most recent training metrics.
 *
 * @returns Latest metrics or null if no metrics exist
 */
export async function getLatestMetrics(): Promise<TrainingMetrics | null> {
  const response = await request<LatestMetricsResponse>('/training/metrics/latest');
  return response.metrics;
}

/**
 * Get all training metrics (convenience function that fetches all available).
 *
 * @returns All available metrics
 */
export async function getAllTrainingMetrics(): Promise<TrainingMetrics[]> {
  // First get total count
  const initial = await getTrainingMetrics(1, 0);
  if (initial.total === 0) {
    return [];
  }

  // Then fetch all
  const all = await getTrainingMetrics(initial.total, 0);
  return all.metrics;
}

/**
 * Get training dashboard data (infrastructure metrics).
 *
 * @returns Dashboard data with workers, games, and models
 */
export async function getDashboardData(): Promise<TrainingDashboardData> {
  return request('/training/dashboard');
}

export { MetricsAPIError };

import type { GameState, AIMoveResponse, LegalMove } from '../types';

const API_BASE = '/api';

class EngineAPIError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
    this.name = 'EngineAPIError';
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }));
    throw new EngineAPIError(response.status, error.code || 'UNKNOWN', error.message);
  }

  return response.json();
}

// Create a new game
export async function createGame(options?: {
  player1_type?: 'human' | 'ai';
  player2_type?: 'human' | 'ai';
  ai_simulations?: number;
}): Promise<{ game_id: string }> {
  return request('/games', {
    method: 'POST',
    body: JSON.stringify(options || {}),
  });
}

// Get current game state
export async function getGameState(gameId: string): Promise<GameState> {
  return request(`/games/${gameId}`);
}

// Make a move
export async function makeMove(gameId: string, move: number): Promise<GameState> {
  return request(`/games/${gameId}/move`, {
    method: 'POST',
    body: JSON.stringify({ move }),
  });
}

// Get AI move
export async function getAIMove(
  gameId: string,
  options?: { simulations?: number; temperature?: number }
): Promise<AIMoveResponse> {
  return request(`/games/${gameId}/ai`, {
    method: 'POST',
    body: JSON.stringify(options || {}),
  });
}

// Get legal moves with details
export async function getLegalMoves(gameId: string): Promise<{ moves: LegalMove[] }> {
  return request(`/games/${gameId}/legal-moves`);
}

// Undo last move
export async function undoMove(gameId: string): Promise<GameState> {
  return request(`/games/${gameId}/undo`, {
    method: 'POST',
  });
}

// Health check
export async function healthCheck(): Promise<{ status: string; version: string }> {
  return request('/health');
}

// WebSocket connection for real-time updates
export function connectWebSocket(
  gameId: string,
  handlers: {
    onState?: (state: GameState) => void;
    onThinking?: (data: { simulations_done: number; simulations_total: number; current_best: string; value: number }) => void;
    onGameOver?: (data: { winner: 0 | 1; reason: string }) => void;
    onError?: (data: { message: string; code: string }) => void;
    onClose?: () => void;
  }
): WebSocket {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/games/${gameId}/ws`;

  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    switch (message.type) {
      case 'state':
        handlers.onState?.(message.data);
        break;
      case 'thinking':
        handlers.onThinking?.(message.data);
        break;
      case 'game_over':
        handlers.onGameOver?.(message.data);
        break;
      case 'error':
        handlers.onError?.(message.data);
        break;
    }
  };

  ws.onclose = () => {
    handlers.onClose?.();
  };

  return ws;
}

// Send move via WebSocket
export function sendMove(ws: WebSocket, move: number): void {
  ws.send(JSON.stringify({ type: 'move', data: { move } }));
}

// Request AI move via WebSocket
export function requestAIMove(ws: WebSocket, simulations?: number): void {
  ws.send(JSON.stringify({ type: 'ai_move', data: { simulations } }));
}

// Request undo via WebSocket
export function requestUndo(ws: WebSocket): void {
  ws.send(JSON.stringify({ type: 'undo' }));
}

export { EngineAPIError };

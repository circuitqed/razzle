/**
 * API functions for online multiplayer games.
 */

import type { GameState } from '../types';

const API_BASE = '/api';

class OnlineAPIError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
    this.name = 'OnlineAPIError';
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
    throw new OnlineAPIError(response.status, error.code || 'UNKNOWN', error.detail || error.message);
  }

  return response.json();
}

// --- Types ---

export interface OnlineOpponentInfo {
  user_id: string;
  display_name: string | null;
  elo_rating?: number;
}

export interface CreateOnlineGameResponse {
  game_id: string;
  join_code: string;
  host_color: number;
  status: string;
}

export interface JoinOnlineGameResponse {
  game_id: string;
  your_color: number;
  opponent: OnlineOpponentInfo | null;
  status: string;
}

export interface OnlineGameStatusResponse {
  game_id: string;
  join_code: string;
  status: string;
  your_color: number;
  opponent: OnlineOpponentInfo | null;
  is_your_turn: boolean;
  winner: number | null;
  game_state: GameState;
}

export interface LeaveOnlineGameResponse {
  status: string;
  winner: number | null;
}

export interface OnlineGameSummary {
  game_id: string;
  join_code: string;
  status: string;
  your_color: number;
  is_your_turn: boolean;
  ply: number;
  opponent_name?: string;
  created_at: string;
  updated_at: string;
  game_mode?: string;
  move_deadline?: string | null;
}

export interface MyOnlineGamesResponse {
  active: OnlineGameSummary[];
  waiting: OnlineGameSummary[];
}

export interface PublicGameSummary {
  game_id: string;
  join_code: string;
  host_display_name: string | null;
  game_mode: string;
  time_control: number | null;
  increment: number;
  days_per_move: number | null;
  created_at: string;
}

export interface QuickMatchResponse {
  action: 'joined' | 'created';
  game_id: string;
  your_color: number;
  join_code: string | null;
}

// --- API Functions ---

/**
 * Create a new online game and get a shareable join code.
 * @param hostColor 0 for blue (player 1), 1 for red (player 2)
 * @param timeControl Total seconds per player (null = untimed)
 * @param increment Seconds added per turn (default 0)
 * @param gameMode 'realtime' or 'correspondence'
 * @param daysPerMove Days per move (correspondence only)
 * @param isPublic Whether the game appears in the public lobby
 */
export async function createOnlineGame(
  hostColor: number = 0,
  timeControl: number | null = null,
  increment: number = 0,
  gameMode: string = 'realtime',
  daysPerMove: number | null = null,
  isPublic: boolean = false,
): Promise<CreateOnlineGameResponse> {
  return request('/games/online', {
    method: 'POST',
    body: JSON.stringify({
      host_color: hostColor,
      time_control: timeControl,
      increment,
      game_mode: gameMode,
      days_per_move: daysPerMove,
      is_public: isPublic,
    }),
  });
}

/**
 * Get public games waiting for opponents.
 */
export async function getPublicLobby(gameMode?: string): Promise<{ games: PublicGameSummary[] }> {
  const params = gameMode ? `?game_mode=${gameMode}` : '';
  return request(`/games/online/lobby${params}`);
}

/**
 * Quick match: join the first available public game, or create a new one.
 */
export async function quickMatch(
  gameMode: string = 'realtime',
  hostColor: number = 0,
  timeControl: number | null = null,
  increment: number = 0,
  daysPerMove: number | null = null,
): Promise<QuickMatchResponse> {
  return request('/games/online/quickmatch', {
    method: 'POST',
    body: JSON.stringify({
      game_mode: gameMode,
      host_color: hostColor,
      time_control: timeControl,
      increment,
      days_per_move: daysPerMove,
    }),
  });
}

/**
 * Join an existing online game using a join code.
 */
export async function joinOnlineGame(joinCode: string): Promise<JoinOnlineGameResponse> {
  return request('/games/online/join', {
    method: 'POST',
    body: JSON.stringify({ join_code: joinCode }),
  });
}

/**
 * Get the current status of an online game.
 */
export async function getOnlineGameStatus(gameId: string): Promise<OnlineGameStatusResponse> {
  return request(`/games/online/${gameId}`);
}

/**
 * Leave/abandon an online game (forfeit).
 */
export async function leaveOnlineGame(gameId: string): Promise<LeaveOnlineGameResponse> {
  return request(`/games/online/${gameId}/leave`, {
    method: 'POST',
  });
}

/**
 * Get all online games for the current user.
 */
export async function getMyOnlineGames(): Promise<MyOnlineGamesResponse> {
  return request('/games/online/mine');
}

// --- WebSocket Types ---

export interface OnlineWebSocketHandlers {
  onState?: (state: GameState & { online_status?: string; your_color?: number; opponent_connected?: boolean }) => void;
  onPlayerJoined?: (data: { user_id: string; display_name: string; color: number }) => void;
  onOpponentDisconnected?: (data: { user_id: string; color: number; grace_period: number }) => void;
  onOpponentReconnected?: (data: { user_id: string; color: number }) => void;
  onGameOver?: (data: { winner: number | null; reason: string }) => void;
  onGameAbandoned?: (data: { abandoning_user_id: string; winner: number | null; reason?: string }) => void;
  onRematchOffered?: (data: { from_color: number }) => void;
  onRematchCreated?: (data: { new_game_id: string }) => void;
  onRematchDeclined?: () => void;
  onError?: (data: { message: string; code: string }) => void;
  onClose?: (code?: number, reason?: string) => void;
  onOpen?: () => void;
  onPong?: () => void;
}

/**
 * Connect to an online game via WebSocket.
 * Handles authentication via cookies automatically.
 */
export function connectOnlineGameWebSocket(
  gameId: string,
  handlers: OnlineWebSocketHandlers
): WebSocket {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/games/${gameId}/ws`;

  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    handlers.onOpen?.();
  };

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    switch (message.type) {
      case 'state':
        handlers.onState?.(message.data);
        break;
      case 'player_joined':
        handlers.onPlayerJoined?.(message.data);
        break;
      case 'opponent_disconnected':
        handlers.onOpponentDisconnected?.(message.data);
        break;
      case 'opponent_reconnected':
        handlers.onOpponentReconnected?.(message.data);
        break;
      case 'game_over':
        handlers.onGameOver?.(message.data);
        break;
      case 'game_abandoned':
        handlers.onGameAbandoned?.(message.data);
        break;
      case 'rematch_offered':
        handlers.onRematchOffered?.(message.data);
        break;
      case 'rematch_created':
        handlers.onRematchCreated?.(message.data);
        break;
      case 'rematch_declined':
        handlers.onRematchDeclined?.();
        break;
      case 'error':
        handlers.onError?.(message.data);
        break;
      case 'pong':
        handlers.onPong?.();
        break;
    }
  };

  ws.onclose = (event) => {
    handlers.onClose?.(event.code, event.reason);
  };

  return ws;
}

/**
 * Send a move in an online game.
 */
export function sendOnlineMove(ws: WebSocket, move: number): void {
  ws.send(JSON.stringify({ type: 'move', data: { move } }));
}

/**
 * Send a ping to keep the connection alive.
 */
export function sendPing(ws: WebSocket): void {
  ws.send(JSON.stringify({ type: 'ping' }));
}

export function sendRematchRequest(ws: WebSocket): void {
  ws.send(JSON.stringify({ type: 'rematch_request' }));
}

export function sendRematchAccept(ws: WebSocket): void {
  ws.send(JSON.stringify({ type: 'rematch_accept' }));
}

export function sendRematchDecline(ws: WebSocket): void {
  ws.send(JSON.stringify({ type: 'rematch_decline' }));
}

/**
 * Send a resign message in an online game.
 */
export function sendResign(ws: WebSocket): void {
  ws.send(JSON.stringify({ type: 'resign' }));
}

export { OnlineAPIError };

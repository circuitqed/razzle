/**
 * Hook for managing online multiplayer game state via WebSocket.
 * Delegates board interaction (selection, click/drag, derived state, history)
 * to the shared useBoardInteraction hook.
 *
 * Sub-moves are buffered client-side by useBoardInteraction.
 * Only complete turns are sent to the server via commitTurn.
 */

import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import type { GameState } from '../types';
import { decodeMove } from '../types';
import * as onlineApi from '../api/online';
import { logger } from '../utils/logger';
import {
  playWinSound,
  playLoseSound,
  playSelectSound,
} from '../utils/sounds';
import { useBoardInteraction } from './useBoardInteraction';

const END_TURN_MOVE = -1;
const PING_INTERVAL = 30000; // 30 seconds
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000, 32000]; // Exponential backoff

interface LastMove {
  from: number;
  to: number;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'reconnecting' | 'disconnected';

export type RematchState = 'none' | 'sent' | 'received';

export interface UseOnlineGameReturn {
  gameState: GameState | null;
  myColor: 0 | 1 | null;
  isMyTurn: boolean;
  opponent: onlineApi.OnlineOpponentInfo | null;
  opponentConnected: boolean;
  connectionStatus: ConnectionStatus;
  onlineStatus: string;
  selectedSquare: number | null;
  isLoading: boolean;
  error: string | null;
  canEndTurn: boolean;
  mustPass: boolean;
  isPassing: boolean;
  lastMove: LastMove | null;
  lastTurnAnimMoves: LastMove[] | undefined;
  rawMoves: number[];
  disconnectWarning: { gracePeriod: number; startTime: number } | null;
  handleSquareClick: (square: number) => void;
  handleDragMove: (from: number, to: number) => void;
  endTurn: () => void;
  cancelPass: () => void;
  leaveGame: () => Promise<void>;
  reconnect: () => void;
  // History navigation
  viewPly: number | null;
  isViewingHistory: boolean;
  effectiveGameState: GameState | null;
  displayLastMove: LastMove | null;
  goToMove: (ply: number) => void;
  goForward: () => void;
  goBack: () => void;
  goToStart: () => void;
  goToEnd: () => void;
  // Rematch
  rematchState: RematchState;
  rematchGameId: string | null;
  requestRematch: () => void;
  acceptRematch: () => void;
  declineRematch: () => void;
}

interface UseOnlineGameOptions {
  gameId: string;
  onGameEnd?: (winner: number | null, reason: string) => void;
  onOpponentJoined?: (opponent: onlineApi.OnlineOpponentInfo) => void;
}

export function useOnlineGame(options: UseOnlineGameOptions): UseOnlineGameReturn {
  const { gameId, onGameEnd, onOpponentJoined } = options;

  // Game state
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [myColor, setMyColor] = useState<0 | 1 | null>(null);
  const [opponent, setOpponent] = useState<onlineApi.OnlineOpponentInfo | null>(null);
  const [opponentConnected, setOpponentConnected] = useState(false);
  const [onlineStatus, setOnlineStatus] = useState<string>('unknown');

  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMove, setLastMove] = useState<LastMove | null>(null);
  const [rawMoves, setRawMoves] = useState<number[]>([]);
  const [disconnectWarning, setDisconnectWarning] = useState<{
    gracePeriod: number;
    startTime: number;
  } | null>(null);
  const [lastTurnAnimMoves, setLastTurnAnimMoves] = useState<LastMove[] | undefined>(undefined);

  // Rematch state
  const [rematchState, setRematchState] = useState<RematchState>('none');
  const [rematchGameId, setRematchGameId] = useState<string | null>(null);

  // Reset rematch state when gameId changes (e.g. after navigating to a rematch game)
  useEffect(() => {
    setRematchState('none');
    setRematchGameId(null);
  }, [gameId]);

  // Connection state
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Generation counter: incremented on each cleanup so stale WS handlers are ignored
  const connectionGenRef = useRef(0);
  const commitInProgressRef = useRef(false);
  const prevRawMovesLenRef = useRef(0);

  // Refs for values used in callbacks to avoid re-creating connect function
  const myColorRef = useRef<0 | 1 | null>(null);
  const onlineStatusRef = useRef<string>('unknown');
  const onGameEndRef = useRef(onGameEnd);
  const onOpponentJoinedRef = useRef(onOpponentJoined);

  // Keep refs in sync
  useEffect(() => {
    myColorRef.current = myColor;
  }, [myColor]);

  useEffect(() => {
    onlineStatusRef.current = onlineStatus;
  }, [onlineStatus]);

  useEffect(() => {
    onGameEndRef.current = onGameEnd;
  }, [onGameEnd]);

  useEffect(() => {
    onOpponentJoinedRef.current = onOpponentJoined;
  }, [onOpponentJoined]);

  // Derived state
  const isMyTurn = useMemo(() => {
    if (!gameState || myColor === null) return false;
    return gameState.current_player === myColor;
  }, [gameState, myColor]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      connectionGenRef.current++;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
    };
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Capture generation at connect-time; all handlers check this to ignore stale connections
    const gen = connectionGenRef.current;
    const isStale = () => gen !== connectionGenRef.current;

    setConnectionStatus('connecting');
    logger.info(`[useOnlineGame] Connecting to game: ${gameId} gen: ${gen}`);

    const ws = onlineApi.connectOnlineGameWebSocket(gameId, {
      onOpen: () => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Connected gen:', gen);
        setConnectionStatus('connected');
        setError(null);
        reconnectAttemptRef.current = 0;

        // Start ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        pingIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            onlineApi.sendPing(wsRef.current);
          }
        }, PING_INTERVAL);
      },

      onState: (data) => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Received state gen:', gen);
        setGameState(data);
        setError(null);
        if (data.your_color !== undefined) {
          setMyColor(data.your_color as 0 | 1);
          myColorRef.current = data.your_color as 0 | 1;
        }
        if (data.opponent_connected !== undefined) {
          setOpponentConnected(data.opponent_connected);
        }
        if (data.online_status) {
          setOnlineStatus(data.online_status);
          onlineStatusRef.current = data.online_status;
        }
        // Sync rawMoves and lastMove from server state
        if (data.moves) {
          // Compute new moves since last state to detect multi-pass turns
          const prevLen = prevRawMovesLenRef.current;
          const newMoves = data.moves.slice(prevLen);
          const nonEndTurns = newMoves.filter((m: number) => m !== END_TURN_MOVE);

          // Set multi-pass animation only for opponent's incremental moves
          // Skip on initial load (prevLen === 0) and our own echoed turns
          if (nonEndTurns.length > 1 && prevLen > 0 && !commitInProgressRef.current) {
            setLastTurnAnimMoves(nonEndTurns.map((m: number) => {
              const { src, dst } = decodeMove(m);
              return { from: src, to: dst };
            }));
          } else {
            setLastTurnAnimMoves(undefined);
          }

          prevRawMovesLenRef.current = data.moves.length;
          setRawMoves(data.moves);
          let derivedLastMove: LastMove | null = null;
          for (let i = data.moves.length - 1; i >= 0; i--) {
            if (data.moves[i] !== END_TURN_MOVE) {
              const { src, dst } = decodeMove(data.moves[i]);
              derivedLastMove = { from: src, to: dst };
              break;
            }
          }
          setLastMove(derivedLastMove);
        }
        commitInProgressRef.current = false;
        setIsLoading(false);
      },

      onPlayerJoined: (data) => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Player joined:', data);
        setOpponentConnected(true);
        setOnlineStatus('playing');
        onlineStatusRef.current = 'playing';
        if (onOpponentJoinedRef.current) {
          onOpponentJoinedRef.current({
            user_id: data.user_id,
            display_name: data.display_name,
            elo_rating: 1000,
          });
        }
        playSelectSound();
      },

      onOpponentDisconnected: (data) => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Opponent disconnected:', data);
        setOpponentConnected(false);
        setDisconnectWarning({
          gracePeriod: data.grace_period,
          startTime: Date.now(),
        });
      },

      onOpponentReconnected: (data) => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Opponent reconnected:', data);
        setOpponentConnected(true);
        setDisconnectWarning(null);
      },

      onGameOver: (data) => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Game over:', data);
        setOnlineStatus('finished');
        onlineStatusRef.current = 'finished';
        setGameState(prev => prev ? { ...prev, status: 'finished' as const, winner: data.winner as 0 | 1 | null } : prev);
        if (data.winner === myColorRef.current) {
          playWinSound();
        } else if (data.winner !== null) {
          playLoseSound();
        }
        onGameEndRef.current?.(data.winner, data.reason);
      },

      onGameAbandoned: (data) => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Game abandoned:', data);
        setOnlineStatus('abandoned');
        onlineStatusRef.current = 'abandoned';
        setDisconnectWarning(null);
        setGameState(prev => prev ? { ...prev, status: 'finished' as const, winner: data.winner as 0 | 1 | null } : prev);
        if (data.winner === myColorRef.current) {
          playWinSound();
        } else if (data.winner !== null) {
          playLoseSound();
        }
        onGameEndRef.current?.(data.winner, data.reason || 'abandoned');
      },

      onRematchOffered: (data) => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Rematch offered:', data);
        if (data.from_color !== myColorRef.current) {
          setRematchState('received');
        }
      },

      onRematchCreated: (data) => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Rematch created:', data);
        setRematchGameId(data.new_game_id);
      },

      onRematchDeclined: () => {
        if (isStale()) return;
        logger.info('[useOnlineGame] Rematch declined');
        setRematchState('none');
      },

      onError: (data) => {
        if (isStale()) return;
        logger.error('[useOnlineGame] WebSocket error:', data);
        if (!commitInProgressRef.current) {
          setError(data.message);
        }
        commitInProgressRef.current = false;
        setIsLoading(false);
      },

      onClose: (code?: number) => {
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        // If generation has advanced, this is a stale connection — don't reconnect
        if (isStale()) {
          logger.info(`[useOnlineGame] Stale connection closed (gen: ${gen} current: ${connectionGenRef.current})`);
          return;
        }

        // If server closed us because a newer connection superseded, don't reconnect
        if (code === 4008) {
          logger.info('[useOnlineGame] Connection superseded by newer connection');
          return;
        }

        logger.info(`[useOnlineGame] Connection closed gen: ${gen} code: ${code}`);

        const status = onlineStatusRef.current;
        if (status === 'playing' || status === 'waiting') {
          const attempt = reconnectAttemptRef.current;
          if (attempt < RECONNECT_DELAYS.length) {
            setConnectionStatus('reconnecting');
            const delay = RECONNECT_DELAYS[attempt];
            logger.info(`[useOnlineGame] Reconnecting in ${delay}ms (attempt ${attempt + 1})`);
            setTimeout(() => {
              reconnectAttemptRef.current++;
              connect();
            }, delay);
          } else {
            setConnectionStatus('disconnected');
            setError('Connection lost. Please refresh the page.');
          }
        } else {
          setConnectionStatus('disconnected');
        }
      },

      onPong: () => {},
    });

    wsRef.current = ws;
  }, [gameId]);

  // Initial connection
  useEffect(() => {
    connect();
    return () => {
      // Bump generation so any handlers from this connection become stale
      connectionGenRef.current++;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  // Load initial game data
  useEffect(() => {
    const loadGameStatus = async () => {
      try {
        const status = await onlineApi.getOnlineGameStatus(gameId);
        setMyColor(status.your_color as 0 | 1);
        setOpponent(status.opponent);
        setOnlineStatus(status.status);
      } catch (err) {
        logger.error('[useOnlineGame] Failed to load game status:', err);
        if (err instanceof onlineApi.OnlineAPIError) {
          setError(err.message);
        }
      }
    };
    loadGameStatus();
  }, [gameId]);

  // Commit a complete turn: send all sub-moves via WebSocket.
  // The server processes them in order and broadcasts the final state.
  const commitTurn = useCallback(
    (moves: number[]) => {
      if (!wsRef.current || connectionStatus !== 'connected') {
        setError('Not connected');
        return;
      }
      // Guard against double-sends (e.g. from duplicate WS connections)
      if (commitInProgressRef.current) {
        logger.info('[useOnlineGame] Ignoring duplicate commitTurn');
        return;
      }
      commitInProgressRef.current = true;

      logger.info('[useOnlineGame] Committing turn:', moves);
      setIsLoading(true);

      // Send complete turn as a single atomic message
      onlineApi.sendOnlineTurn(wsRef.current, moves);
      // State will be updated by onState handler when server processes the turn
    },
    [connectionStatus]
  );

  // Board interaction hook
  const {
    selectedSquare, handleSquareClick, handleDragMove,
    endTurn, cancelPass, canEndTurn, isPassing, mustPass,
    viewPly, isViewingHistory, effectiveGameState, displayLastMove,
    goToMove, goForward, goBack, goToStart, goToEnd,
  } = useBoardInteraction({
    gameState,
    rawMoves,
    lastMove,
    isInteractionEnabled: isMyTurn,
    isLoading,
    commitTurn,
  });

  // Leave/abandon game
  const leaveGame = useCallback(async () => {
    try {
      await onlineApi.leaveOnlineGame(gameId);
    } catch (err) {
      logger.error('[useOnlineGame] Failed to leave game:', err);
      throw err;
    }
  }, [gameId]);

  // Manual reconnect
  const reconnect = useCallback(() => {
    reconnectAttemptRef.current = 0;
    // Bump generation so the old connection's onClose won't trigger auto-reconnect
    connectionGenRef.current++;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    connect();
  }, [connect]);

  // Rematch actions
  const requestRematch = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    onlineApi.sendRematchRequest(wsRef.current);
    setRematchState('sent');
  }, []);

  const acceptRematch = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    onlineApi.sendRematchAccept(wsRef.current);
  }, []);

  const declineRematch = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    onlineApi.sendRematchDecline(wsRef.current);
    setRematchState('none');
  }, []);

  return {
    gameState: effectiveGameState,
    myColor,
    isMyTurn,
    opponent,
    opponentConnected,
    connectionStatus,
    onlineStatus,
    selectedSquare,
    isLoading,
    error,
    canEndTurn,
    mustPass,
    isPassing,
    lastMove: displayLastMove,
    lastTurnAnimMoves,
    rawMoves,
    disconnectWarning,
    handleSquareClick,
    handleDragMove,
    endTurn,
    cancelPass,
    leaveGame,
    reconnect,
    viewPly,
    isViewingHistory,
    effectiveGameState,
    displayLastMove,
    goToMove,
    goForward,
    goBack,
    goToStart,
    goToEnd,
    rematchState,
    rematchGameId,
    requestRematch,
    acceptRematch,
    declineRematch,
  };
}

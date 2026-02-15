/**
 * Hook for managing online multiplayer game state via WebSocket.
 */

import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import type { GameState, Player } from '../types';
import { encodeMove, decodeMove, TOTAL_SQUARES } from '../types';
import * as onlineApi from '../api/online';
import { logger } from '../utils/logger';
import {
  playMoveSound,
  playPassSound,
  playEndTurnSound,
  playWinSound,
  playLoseSound,
  playSelectSound,
} from '../utils/sounds';

const END_TURN_MOVE = -1;
const PING_INTERVAL = 30000; // 30 seconds
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000, 32000]; // Exponential backoff

interface LastMove {
  from: number;
  to: number;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'reconnecting' | 'disconnected';

export interface UseOnlineGameReturn {
  gameState: GameState | null;
  myColor: 0 | 1 | null;
  isMyTurn: boolean;
  opponent: onlineApi.OnlineOpponentInfo | null;
  opponentConnected: boolean;
  connectionStatus: ConnectionStatus;
  selectedSquare: number | null;
  isLoading: boolean;
  error: string | null;
  canEndTurn: boolean;
  mustPass: boolean;
  lastMove: LastMove | null;
  rawMoves: number[];
  disconnectWarning: { gracePeriod: number; startTime: number } | null;
  makeMove: (move: number) => void;
  handleSquareClick: (square: number) => void;
  handleDragMove: (from: number, to: number) => void;
  endTurn: () => void;
  leaveGame: () => Promise<void>;
  reconnect: () => void;
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
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMove, setLastMove] = useState<LastMove | null>(null);
  const [rawMoves, setRawMoves] = useState<number[]>([]);
  const [disconnectWarning, setDisconnectWarning] = useState<{
    gracePeriod: number;
    startTime: number;
  } | null>(null);

  // Connection state
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  const canEndTurn = useMemo(() => {
    if (!gameState || !isMyTurn) return false;
    return gameState.legal_moves.includes(END_TURN_MOVE);
  }, [gameState, isMyTurn]);

  const mustPass = useMemo(() => {
    if (!gameState || !isMyTurn || gameState.has_passed) return false;

    const ballBitboard =
      gameState.current_player === 0 ? gameState.board.p1_ball : gameState.board.p2_ball;

    let ballSquare = -1;
    for (let i = 0; i < TOTAL_SQUARES; i++) {
      if ((BigInt(ballBitboard) & (BigInt(1) << BigInt(i))) !== BigInt(0)) {
        ballSquare = i;
        break;
      }
    }

    if (ballSquare === -1) return false;

    const realMoves = gameState.legal_moves.filter((m) => m !== END_TURN_MOVE);
    if (realMoves.length === 0) return false;

    return realMoves.every((move) => {
      const { src } = decodeMove(move);
      return src === ballSquare;
    });
  }, [gameState, isMyTurn]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
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

    setConnectionStatus('connecting');
    logger.info('[useOnlineGame] Connecting to game:', gameId);

    const ws = onlineApi.connectOnlineGameWebSocket(gameId, {
      onOpen: () => {
        logger.info('[useOnlineGame] Connected');
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
        logger.info('[useOnlineGame] Received state:', data);
        setGameState(data);
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
        setIsLoading(false);
      },

      onPlayerJoined: (data) => {
        logger.info('[useOnlineGame] Player joined:', data);
        setOpponentConnected(true);
        setOnlineStatus('playing');
        onlineStatusRef.current = 'playing';
        if (onOpponentJoinedRef.current) {
          onOpponentJoinedRef.current({
            user_id: data.user_id,
            display_name: data.display_name,
            elo_rating: 1000, // Will be updated from game status
          });
        }
        // Play a sound to notify
        playSelectSound();
      },

      onOpponentDisconnected: (data) => {
        logger.info('[useOnlineGame] Opponent disconnected:', data);
        setOpponentConnected(false);
        setDisconnectWarning({
          gracePeriod: data.grace_period,
          startTime: Date.now(),
        });
      },

      onOpponentReconnected: (data) => {
        logger.info('[useOnlineGame] Opponent reconnected:', data);
        setOpponentConnected(true);
        setDisconnectWarning(null);
      },

      onGameOver: (data) => {
        logger.info('[useOnlineGame] Game over:', data);
        setOnlineStatus('finished');
        onlineStatusRef.current = 'finished';
        if (data.winner === myColorRef.current) {
          playWinSound();
        } else if (data.winner !== null) {
          playLoseSound();
        }
        onGameEndRef.current?.(data.winner, data.reason);
      },

      onGameAbandoned: (data) => {
        logger.info('[useOnlineGame] Game abandoned:', data);
        setOnlineStatus('abandoned');
        onlineStatusRef.current = 'abandoned';
        setDisconnectWarning(null);
        if (data.winner === myColorRef.current) {
          playWinSound();
        } else if (data.winner !== null) {
          playLoseSound();
        }
        onGameEndRef.current?.(data.winner, data.reason || 'abandoned');
      },

      onError: (data) => {
        logger.error('[useOnlineGame] WebSocket error:', data);
        setError(data.message);
        setIsLoading(false);
      },

      onClose: () => {
        logger.info('[useOnlineGame] Connection closed');
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        // Only try to reconnect if game is still active
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

      onPong: () => {
        // Keep-alive acknowledged
      },
    });

    wsRef.current = ws;
  }, [gameId]); // Only reconnect if gameId changes

  // Initial connection
  useEffect(() => {
    connect();
    return () => {
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
        // Don't set game state here - WebSocket will send it
      } catch (err) {
        logger.error('[useOnlineGame] Failed to load game status:', err);
        if (err instanceof onlineApi.OnlineAPIError) {
          setError(err.message);
        }
      }
    };
    loadGameStatus();
  }, [gameId]);

  // Make a move
  const makeMove = useCallback(
    (move: number) => {
      if (!wsRef.current || connectionStatus !== 'connected') {
        setError('Not connected');
        return;
      }
      if (!isMyTurn) {
        setError('Not your turn');
        return;
      }

      logger.info('[useOnlineGame] Sending move:', move);
      setIsLoading(true);
      onlineApi.sendOnlineMove(wsRef.current, move);

      // Record move locally (will be confirmed by server state update)
      if (move === END_TURN_MOVE) {
        setRawMoves((prev) => [...prev, move]);
        playEndTurnSound();
      } else {
        const { src, dst } = decodeMove(move);
        setLastMove({ from: src, to: dst });
        setRawMoves((prev) => [...prev, move]);
      }
    },
    [connectionStatus, isMyTurn]
  );

  // Handle square click
  const handleSquareClick = useCallback(
    (square: number) => {
      if (!gameState || gameState.status !== 'playing' || isLoading) return;
      if (!isMyTurn) return;

      const { board, legal_moves, current_player } = gameState;

      // Check if clicking on own piece or ball
      const isOwnPiece = (player: Player, sq: number) => {
        const pieces = BigInt(player === 0 ? board.p1_pieces : board.p2_pieces);
        const ball = BigInt(player === 0 ? board.p1_ball : board.p2_ball);
        const mask = BigInt(1) << BigInt(sq);
        return ((pieces | ball) & mask) !== BigInt(0);
      };

      // If a piece is selected, check if this is a valid move destination
      if (selectedSquare !== null) {
        const moveEncoded = encodeMove(selectedSquare, square);
        if (legal_moves.includes(moveEncoded)) {
          // Check if this is a pass
          const ballBitboard =
            current_player === 0 ? board.p1_ball : board.p2_ball;
          const isBallSquare =
            (BigInt(ballBitboard) & (BigInt(1) << BigInt(selectedSquare))) !== BigInt(0);

          makeMove(moveEncoded);

          if (isBallSquare) {
            // Keep selection on new position for pass chains
            setSelectedSquare(square);
            playPassSound();
          } else {
            setSelectedSquare(null);
            playMoveSound();
          }
          return;
        }
      }

      // Check if clicking on own piece to select/deselect
      if (isOwnPiece(current_player, square)) {
        const newSelection = selectedSquare === square ? null : square;
        setSelectedSquare(newSelection);
        if (newSelection !== null) {
          playSelectSound();
        }
        return;
      }

      // Clicking elsewhere - deselect
      if (selectedSquare !== null) {
        setSelectedSquare(null);
      }
    },
    [gameState, selectedSquare, isMyTurn, isLoading, makeMove]
  );

  // Handle drag-and-drop move
  const handleDragMove = useCallback(
    (from: number, to: number) => {
      if (!gameState || gameState.status !== 'playing' || isLoading) return;
      if (!isMyTurn) return;

      const moveEncoded = encodeMove(from, to);
      if (!gameState.legal_moves.includes(moveEncoded)) return;

      const { board, current_player } = gameState;
      const ballBitboard =
        current_player === 0 ? board.p1_ball : board.p2_ball;
      const isBallSquare =
        (BigInt(ballBitboard) & (BigInt(1) << BigInt(from))) !== BigInt(0);

      makeMove(moveEncoded);

      if (isBallSquare) {
        setSelectedSquare(to);
        playPassSound();
      } else {
        setSelectedSquare(null);
        playMoveSound();
      }
    },
    [gameState, isMyTurn, isLoading, makeMove]
  );

  // End turn
  const endTurn = useCallback(() => {
    if (!canEndTurn) return;
    makeMove(END_TURN_MOVE);
    setSelectedSquare(null);
  }, [canEndTurn, makeMove]);

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
    if (wsRef.current) {
      wsRef.current.close();
    }
    connect();
  }, [connect]);

  return {
    gameState,
    myColor,
    isMyTurn,
    opponent,
    opponentConnected,
    connectionStatus,
    selectedSquare,
    isLoading,
    error,
    canEndTurn,
    mustPass,
    lastMove,
    rawMoves,
    disconnectWarning,
    makeMove,
    handleSquareClick,
    handleDragMove,
    endTurn,
    leaveGame,
    reconnect,
  };
}

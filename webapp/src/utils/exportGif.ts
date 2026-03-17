/**
 * Export a game replay as an animated GIF.
 *
 * Renders each turn-boundary position as an SVG string,
 * rasterizes to canvas, and encodes frames with gif.js.
 */
import GIF from 'gif.js';
import { BOARD_COLS, BOARD_ROWS, hasPiece } from '../types';
import type { ReplayState } from './replay';

const SQUARE_SIZE = 50;
const BOARD_WIDTH = BOARD_COLS * SQUARE_SIZE;
const BOARD_HEIGHT = BOARD_ROWS * SQUARE_SIZE;
const LABEL_PAD_LEFT = 14;
const LABEL_PAD_BOTTOM = 14;
const GIF_WIDTH = 400;
const GIF_HEIGHT = Math.round(
  GIF_WIDTH * (BOARD_HEIGHT + LABEL_PAD_BOTTOM) / (BOARD_WIDTH + LABEL_PAD_LEFT)
);

// Colors matching Board.tsx
const LIGHT_SQ = '#f0d9b5';
const DARK_SQ = '#b58863';
const GOAL_BLUE = '#93c5fd';
const GOAL_RED = '#fca5a5';
const P1_COLOR = '#3b82f6';
const P2_COLOR = '#ef4444';
const BALL_FILL = '#fbbf24';
const BALL_STROKE = '#92400e';
const GRID_STROKE = '#374151';
const LABEL_COLOR = '#9ca3af';
const BG_COLOR = '#111827'; // matches gray-900

function renderBoardSvg(state: ReplayState): string {
  const parts: string[] = [];

  const viewBoxW = BOARD_WIDTH + LABEL_PAD_LEFT;
  const viewBoxH = BOARD_HEIGHT + LABEL_PAD_BOTTOM;

  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${GIF_WIDTH}" height="${GIF_HEIGHT}" viewBox="${-LABEL_PAD_LEFT} 0 ${viewBoxW} ${viewBoxH}">`
  );

  // Background fill for label areas
  parts.push(
    `<rect x="${-LABEL_PAD_LEFT}" y="0" width="${viewBoxW}" height="${viewBoxH}" fill="${BG_COLOR}" />`
  );

  // Draw squares and pieces
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const square = row * BOARD_COLS + col;
      const x = col * SQUARE_SIZE;
      const y = (BOARD_ROWS - 1 - row) * SQUARE_SIZE;

      // Square color
      const isLight = (row + col) % 2 === 0;
      let bgColor: string;
      if (row === 0) {
        bgColor = GOAL_BLUE;
      } else if (row === BOARD_ROWS - 1) {
        bgColor = GOAL_RED;
      } else {
        bgColor = isLight ? LIGHT_SQ : DARK_SQ;
      }

      parts.push(
        `<rect x="${x}" y="${y}" width="${SQUARE_SIZE}" height="${SQUARE_SIZE}" fill="${bgColor}" stroke="${GRID_STROKE}" stroke-width="0.5" />`
      );

      // Pieces
      const hasP1Piece = hasPiece(state.board.p1_pieces, square);
      const hasP2Piece = hasPiece(state.board.p2_pieces, square);
      const hasP1Ball = hasPiece(state.board.p1_ball, square);
      const hasP2Ball = hasPiece(state.board.p2_ball, square);

      if (hasP1Piece || hasP2Piece) {
        const color = hasP1Piece ? P1_COLOR : P2_COLOR;
        // Diamond: relative to (x, y) with SQUARE_SIZE=50 → same points as Piece.tsx
        const cx = x + 25;
        const cy = y + 25;
        parts.push(
          `<polygon points="${cx},${cy - 17} ${cx + 17},${cy} ${cx},${cy + 17} ${cx - 17},${cy}" fill="${color}" stroke="#1f2937" stroke-width="1.5" />`
        );

        // Ball
        if (hasP1Ball || hasP2Ball) {
          parts.push(
            `<circle cx="${cx}" cy="${cy}" r="8" fill="${BALL_FILL}" stroke="${BALL_STROKE}" stroke-width="1.5" />`
          );
        }
      }
    }
  }

  // File labels (a-g)
  for (let col = 0; col < BOARD_COLS; col++) {
    const label = String.fromCharCode('a'.charCodeAt(0) + col);
    const x = col * SQUARE_SIZE + SQUARE_SIZE / 2;
    const y = BOARD_HEIGHT + LABEL_PAD_BOTTOM - 3;
    parts.push(
      `<text x="${x}" y="${y}" text-anchor="middle" font-size="10" fill="${LABEL_COLOR}" font-family="sans-serif">${label}</text>`
    );
  }

  // Rank labels (1-8)
  for (let row = 0; row < BOARD_ROWS; row++) {
    const label = row + 1;
    const x = -3;
    const y = (BOARD_ROWS - 1 - row) * SQUARE_SIZE + SQUARE_SIZE / 2 + 4;
    parts.push(
      `<text x="${x}" y="${y}" text-anchor="end" font-size="10" fill="${LABEL_COLOR}" font-family="sans-serif">${label}</text>`
    );
  }

  parts.push('</svg>');
  return parts.join('');
}

function svgToCanvas(
  svgString: string,
  width: number,
  height: number
): Promise<HTMLCanvasElement> {
  return new Promise((resolve, reject) => {
    const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0, width, height);
      URL.revokeObjectURL(url);
      resolve(canvas);
    };
    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(err);
    };
    img.src = url;
  });
}

/**
 * Export a game replay as an animated GIF.
 *
 * @param positions     All reconstructed positions (one per ply, including initial)
 * @param turnBoundaries Ply indices of turn boundaries (player changes)
 * @param onProgress    Callback with progress 0-1
 * @returns             GIF as a Blob
 */
export async function exportGameGif(
  positions: ReplayState[],
  turnBoundaries: number[],
  onProgress: (pct: number) => void
): Promise<Blob> {
  const gif = new GIF({
    workers: 2,
    quality: 10,
    width: GIF_WIDTH,
    height: GIF_HEIGHT,
    workerScript: '/gif.worker.js',
  });

  // Render each turn boundary position as a frame
  const total = turnBoundaries.length;
  for (let i = 0; i < total; i++) {
    const ply = turnBoundaries[i];
    const state = positions[ply];
    if (!state) continue;

    const svgString = renderBoardSvg(state);
    const canvas = await svgToCanvas(svgString, GIF_WIDTH, GIF_HEIGHT);

    // Last frame holds longer
    const delay = i === total - 1 ? 3000 : 1000;
    gif.addFrame(canvas, { delay, copy: true });

    onProgress((i + 1) / total * 0.8); // 80% for rendering, 20% for encoding
  }

  return new Promise<Blob>((resolve, reject) => {
    gif.on('progress', (p: number) => {
      onProgress(0.8 + p * 0.2);
    });
    gif.on('finished', (blob: Blob) => {
      resolve(blob);
    });
    gif.on('abort', () => {
      reject(new Error('GIF encoding aborted'));
    });
    gif.render();
  });
}

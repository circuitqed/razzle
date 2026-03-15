/**
 * Opening book for instant move lookup in known positions.
 *
 * Loads a JSON book generated from training games and provides
 * move lookup without needing MCTS search.
 */

import type { EngineState } from './state';

interface BookEntry {
  /** [action_index, count, win_rate][] — sorted by count descending */
  moves: [number, number, number][];
  total: number;
}

interface BookData {
  metadata: {
    model_range?: string;
    games?: number;
    min_visits?: number;
    depth?: number;
    generated_at?: string;
  };
  positions: Record<string, BookEntry>;
}

/**
 * Compute position key matching the Python format:
 * "{pieces[0]}:{balls[0]}:{pieces[1]}:{balls[1]}:{current_player}:{touched_mask}:{has_passed}"
 *
 * BigInt values are converted to decimal strings (matching Python int.__str__).
 */
function stateKey(state: EngineState): string {
  return (
    `${state.pieces[0]}:${state.balls[0]}:` +
    `${state.pieces[1]}:${state.balls[1]}:` +
    `${state.currentPlayer}:${state.touchedMask}:${state.hasPassed ? 1 : 0}`
  );
}

export class OpeningBook {
  private positions: Record<string, BookEntry>;
  public readonly size: number;

  private constructor(data: BookData) {
    this.positions = data.positions;
    this.size = Object.keys(this.positions).length;
  }

  /** Load opening book from a URL (e.g. /api/opening-book). */
  static async load(url: string): Promise<OpeningBook> {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to load opening book: ${resp.status}`);
    }
    const data: BookData = await resp.json();
    return new OpeningBook(data);
  }

  /** Load opening book from pre-parsed JSON data. */
  static fromData(data: BookData): OpeningBook {
    return new OpeningBook(data);
  }

  /**
   * Look up the most popular move for a position.
   * Returns action index, or null if position not in book.
   */
  lookup(state: EngineState): number | null {
    const entry = this.positions[stateKey(state)];
    if (!entry) return null;
    return entry.moves[0][0];
  }

  /**
   * Sample a move from the book distribution with temperature.
   * 0 = always pick most popular, 1 = proportional to visits, >1 = more uniform.
   */
  sample(state: EngineState, temperature: number = 1.0): number | null {
    const entry = this.positions[stateKey(state)];
    if (!entry) return null;

    const moves = entry.moves;
    if (temperature === 0 || moves.length === 1) return moves[0][0];

    // Apply temperature to visit counts
    const counts = moves.map(m => m[1]);
    let weights: number[];
    if (temperature !== 1.0) {
      const maxCount = Math.max(...counts);
      const logCounts = counts.map(c => Math.log(c / maxCount) / temperature);
      const maxLog = Math.max(...logCounts);
      weights = logCounts.map(lc => Math.exp(lc - maxLog));
    } else {
      weights = counts;
    }

    const total = weights.reduce((a, b) => a + b, 0);
    let r = Math.random() * total;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return moves[i][0];
    }
    return moves[moves.length - 1][0];
  }

  /**
   * Look up the move distribution for a position.
   * Returns [(action, probability), ...] or null if not in book.
   */
  lookupDistribution(state: EngineState): Array<[number, number]> | null {
    const entry = this.positions[stateKey(state)];
    if (!entry) return null;
    return entry.moves.map(([action, count]) => [action, count / entry.total]);
  }

  /**
   * Look up raw book data for a position (for the opening explorer UI).
   * Returns moves with raw counts and win rates, or null if not in book.
   */
  lookupRaw(state: EngineState): { moves: [number, number, number][]; total: number } | null {
    const entry = this.positions[stateKey(state)];
    if (!entry) return null;
    return { moves: entry.moves, total: entry.total };
  }
}

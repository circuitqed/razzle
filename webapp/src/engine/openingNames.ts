/**
 * Named openings for the Razzle Dazzle opening explorer.
 *
 * The board is symmetric around column d (a↔g, b↔f, c↔e).
 * Each named opening is stored in canonical form (f-side);
 * matching checks both canonical and mirror automatically.
 */

interface NamedOpening {
  name: string;
  /** Canonical turn labels to match, e.g. ["f1-e3", "b8-c6"] */
  moves: string[];
}

/** Mirror a column letter across the d-axis. */
function mirrorCol(c: string): string {
  const map: Record<string, string> = { a: 'g', b: 'f', c: 'e', d: 'd', e: 'c', f: 'b', g: 'a' };
  return map[c] || c;
}

/** Mirror a square (e.g. "f1" → "b1"). */
function mirrorSquare(sq: string): string {
  return mirrorCol(sq[0]) + sq.slice(1);
}

/** Mirror a turn label (e.g. "f1-e3" → "b1-c3", "d1-c1-b1" → "d1-e1-f1"). */
function mirrorLabel(label: string): string {
  return label.split('-').map(mirrorSquare).join('-');
}

/**
 * Named openings, sorted longest-first so the most specific match wins.
 * All moves are in canonical (f-side) form.
 */
const NAMED_OPENINGS: NamedOpening[] = [
  // 3-ply: Green Opening + defense + d5 Advance
  { name: "Green Opening: Nora Defense, d5 Advance", moves: ["f1-e3", "e8-d6", "e3-d5"] },
  { name: "Green Opening: Ian Defense, d5 Advance", moves: ["f1-e3", "b8-c6", "e3-d5"] },
  { name: "Green Opening: Caroline Defense, d5 Advance", moves: ["f1-e3", "f8-e6", "e3-d5"] },

  // 2-ply: Green Opening + Red defense
  { name: "Green Opening: Nora Defense", moves: ["f1-e3", "e8-d6"] },
  { name: "Green Opening: Ian Defense", moves: ["f1-e3", "b8-c6"] },
  { name: "Green Opening: Caroline Defense", moves: ["f1-e3", "f8-e6"] },

  // 1-ply: Blue openings
  { name: "Green Opening", moves: ["f1-e3"] },
  { name: "Schuster System", moves: ["e1-d3"] },
  { name: "The Larry", moves: ["c1-a2"] },
];

/**
 * Find the most specific named opening matching the given turn labels.
 * Handles left-right symmetry automatically.
 * Returns null if no named opening matches.
 */
export function getOpeningName(turnLabels: string[]): string | null {
  for (const opening of NAMED_OPENINGS) {
    if (opening.moves.length > turnLabels.length) continue;

    // Try canonical match
    let match = true;
    for (let i = 0; i < opening.moves.length; i++) {
      if (turnLabels[i] !== opening.moves[i]) { match = false; break; }
    }
    if (match) return opening.name;

    // Try mirror match
    match = true;
    for (let i = 0; i < opening.moves.length; i++) {
      if (turnLabels[i] !== mirrorLabel(opening.moves[i])) { match = false; break; }
    }
    if (match) return opening.name;
  }
  return null;
}

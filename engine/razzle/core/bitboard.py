"""
Bitboard utilities for Razzle Dazzle.

Board layout (8 rows x 7 cols = 56 squares, fits in 64-bit int):

  8 | 49 50 51 52 53 54 55
  7 | 42 43 44 45 46 47 48
  6 | 35 36 37 38 39 40 41
  5 | 28 29 30 31 32 33 34
  4 | 21 22 23 24 25 26 27
  3 | 14 15 16 17 18 19 20
  2 |  7  8  9 10 11 12 13
  1 |  0  1  2  3  4  5  6
    +---------------------
       a  b  c  d  e  f  g

Square index = row * 7 + col (row 0 = rank 1, col 0 = file a)
"""

from typing import Iterator

# Board dimensions
ROWS = 8
COLS = 7
NUM_SQUARES = ROWS * COLS  # 56

# Mask for valid squares (bits 0-55)
VALID_MASK = (1 << NUM_SQUARES) - 1

# Starting positions
# Player 1 (bottom): pieces on b1-f1, ball on d1
# Player 2 (top): pieces on b8-f8, ball on d8
P1_START_PIECES = 0b0111110  # bits 1-5 (b1-f1)
P1_START_BALL = 0b0001000    # bit 3 (d1)
P2_START_PIECES = P1_START_PIECES << (7 * 7)  # row 8
P2_START_BALL = P1_START_BALL << (7 * 7)      # d8

# Goal rows (where ball must reach to win)
ROW_1_MASK = 0b1111111  # bits 0-6
ROW_8_MASK = ROW_1_MASK << (7 * 7)  # bits 49-55

# Precomputed tables (initialized at module load)
KNIGHT_ATTACKS: list[int] = [0] * NUM_SQUARES
RAY_MASKS: list[list[int]] = [[0] * 8 for _ in range(NUM_SQUARES)]  # [sq][dir]
BETWEEN: list[list[int]] = [[0] * NUM_SQUARES for _ in range(NUM_SQUARES)]  # [from][to]

# Direction offsets for rays (passes): N, S, E, W, NE, NW, SE, SW
RAY_DIRS = [
    7,    # N  (+1 row)
    -7,   # S  (-1 row)
    1,    # E  (+1 col)
    -1,   # W  (-1 col)
    8,    # NE (+1 row, +1 col)
    6,    # NW (+1 row, -1 col)
    -6,   # SE (-1 row, +1 col)
    -8,   # SW (-1 row, -1 col)
]

# Knight move offsets (row_delta, col_delta)
KNIGHT_DELTAS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1)
]


def sq_to_rowcol(sq: int) -> tuple[int, int]:
    """Convert square index to (row, col)."""
    return sq // COLS, sq % COLS


def rowcol_to_sq(row: int, col: int) -> int:
    """Convert (row, col) to square index."""
    return row * COLS + col


def sq_to_algebraic(sq: int) -> str:
    """Convert square index to algebraic notation (e.g., 'd1')."""
    row, col = sq_to_rowcol(sq)
    return chr(ord('a') + col) + str(row + 1)


def algebraic_to_sq(s: str) -> int:
    """Convert algebraic notation to square index."""
    col = ord(s[0].lower()) - ord('a')
    row = int(s[1]) - 1
    return rowcol_to_sq(row, col)


def is_valid_sq(row: int, col: int) -> bool:
    """Check if (row, col) is on the board."""
    return 0 <= row < ROWS and 0 <= col < COLS


def bit(sq: int) -> int:
    """Return bitboard with single bit set at square."""
    return 1 << sq


def popcount(bb: int) -> int:
    """Count number of set bits."""
    return bin(bb).count('1')


def lsb(bb: int) -> int:
    """Return index of least significant bit (or -1 if empty)."""
    bb = int(bb)  # Handle numpy int64
    if bb == 0:
        return -1
    return (bb & -bb).bit_length() - 1


def iter_bits(bb: int) -> Iterator[int]:
    """Iterate over indices of set bits."""
    bb = int(bb)  # Handle numpy int64
    while bb:
        sq = lsb(bb)
        yield sq
        bb &= bb - 1  # Clear LSB


def bb_to_squares(bb: int) -> list[int]:
    """Convert bitboard to list of square indices."""
    return list(iter_bits(bb))


def print_bitboard(bb: int, label: str = "") -> None:
    """Print bitboard in readable format."""
    if label:
        print(f"{label}:")
    for row in range(ROWS - 1, -1, -1):
        rank = str(row + 1) + " |"
        for col in range(COLS):
            sq = rowcol_to_sq(row, col)
            rank += " 1" if bb & bit(sq) else " ."
        print(rank)
    print("   +" + "-" * (COLS * 2))
    print("    " + " ".join("abcdefg"))


def _init_knight_attacks() -> None:
    """Precompute knight attack bitboards for all squares."""
    for sq in range(NUM_SQUARES):
        row, col = sq_to_rowcol(sq)
        attacks = 0
        for dr, dc in KNIGHT_DELTAS:
            r, c = row + dr, col + dc
            if is_valid_sq(r, c):
                attacks |= bit(rowcol_to_sq(r, c))
        KNIGHT_ATTACKS[sq] = attacks


def _init_ray_masks() -> None:
    """Precompute ray masks for all squares and directions."""
    for sq in range(NUM_SQUARES):
        row, col = sq_to_rowcol(sq)
        for dir_idx, (dr, dc) in enumerate([
            (1, 0), (-1, 0), (0, 1), (0, -1),  # N, S, E, W
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # NE, NW, SE, SW
        ]):
            mask = 0
            r, c = row + dr, col + dc
            while is_valid_sq(r, c):
                mask |= bit(rowcol_to_sq(r, c))
                r += dr
                c += dc
            RAY_MASKS[sq][dir_idx] = mask


def _init_between() -> None:
    """Precompute squares between any two squares on same line."""
    for sq1 in range(NUM_SQUARES):
        r1, c1 = sq_to_rowcol(sq1)
        for sq2 in range(NUM_SQUARES):
            if sq1 == sq2:
                continue
            r2, c2 = sq_to_rowcol(sq2)
            dr = 0 if r2 == r1 else (1 if r2 > r1 else -1)
            dc = 0 if c2 == c1 else (1 if c2 > c1 else -1)

            # Check if on same line (horizontal, vertical, or diagonal)
            if dr == 0 and dc == 0:
                continue
            if dr != 0 and dc != 0 and abs(r2 - r1) != abs(c2 - c1):
                continue
            if dr == 0 and r2 != r1:
                continue
            if dc == 0 and c2 != c1:
                continue

            # Build mask of squares between
            mask = 0
            r, c = r1 + dr, c1 + dc
            while (r, c) != (r2, c2) and is_valid_sq(r, c):
                mask |= bit(rowcol_to_sq(r, c))
                r += dr
                c += dc
            BETWEEN[sq1][sq2] = mask


def get_ray_to(from_sq: int, to_sq: int, occupied: int) -> int | None:
    """
    Get the first piece encountered on ray from from_sq toward to_sq.
    Returns square index or None if no piece found or not on same line.
    """
    r1, c1 = sq_to_rowcol(from_sq)
    r2, c2 = sq_to_rowcol(to_sq)

    dr = 0 if r2 == r1 else (1 if r2 > r1 else -1)
    dc = 0 if c2 == c1 else (1 if c2 > c1 else -1)

    if dr == 0 and dc == 0:
        return None

    r, c = r1 + dr, c1 + dc
    while is_valid_sq(r, c):
        sq = rowcol_to_sq(r, c)
        if occupied & bit(sq):
            return sq
        r += dr
        c += dc

    return None


# Initialize lookup tables at module load
_init_knight_attacks()
_init_ray_masks()
_init_between()

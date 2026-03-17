"""Generate an animated GIF of a Razzle Dazzle game replay."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image, ImageDraw, ImageFont
from razzle.core.state import GameState

# Board dimensions
COLS = 7  # a-g
ROWS = 8  # 1-8

# Rendering config
CELL_SIZE = 72
BOARD_MARGIN = 32
LABEL_SPACE = 24
PIECE_RADIUS = 25
BALL_RADIUS = 11

# Sharp, saturated colors — no alpha blending on main elements
BG_COLOR = (18, 20, 30)
LIGHT_SQUARE = (232, 212, 182)
DARK_SQUARE = (180, 142, 105)
P1_COLOR = (30, 110, 235)       # Vivid blue
P1_OUTLINE = (20, 70, 180)
P2_COLOR = (225, 40, 40)        # Vivid red
P2_OUTLINE = (175, 25, 25)
BALL_COLOR = (255, 195, 30)     # Bright yellow-orange
BALL_OUTLINE = (220, 155, 10)
TEXT_COLOR = (210, 210, 220)
WINNER_BLUE = (60, 160, 255)
WINNER_RED = (255, 70, 70)

# Goal row colors (applied directly, no alpha)
GOAL_P1_LIGHT = (235, 200, 190)  # Red-tinted light square
GOAL_P1_DARK = (195, 140, 120)   # Red-tinted dark square
GOAL_P2_LIGHT = (200, 215, 240)  # Blue-tinted light square
GOAL_P2_DARK = (155, 160, 185)   # Blue-tinted dark square

# Highlight color for last move (applied directly)
HIGHLIGHT_LIGHT = (245, 240, 160)
HIGHLIGHT_DARK = (210, 200, 120)

COL_LABELS = 'abcdefg'
ROW_LABELS = '12345678'


def square_to_rc(sq):
    """Convert square index (0-55) to (row, col) where row 0 is rank 1."""
    return sq // COLS, sq % COLS


def rc_to_pixel(row, col):
    """Convert (row, col) to pixel center. Row 0 (rank 1) at bottom."""
    visual_row = ROWS - 1 - row
    x = BOARD_MARGIN + LABEL_SPACE + col * CELL_SIZE + CELL_SIZE // 2
    y = BOARD_MARGIN + visual_row * CELL_SIZE + CELL_SIZE // 2
    return x, y


def get_pieces(state):
    """Extract absolute p1/p2 piece positions from state."""
    if state.current_player == 0:
        p1_bb, p1_ball_bb = state.my_pieces, state.my_ball
        p2_bb, p2_ball_bb = state.opp_pieces, state.opp_ball
    else:
        p1_bb, p1_ball_bb = state.opp_pieces, state.opp_ball
        p2_bb, p2_ball_bb = state.my_pieces, state.my_ball

    p1, p1_ball, p2, p2_ball = [], None, [], None
    for sq in range(56):
        mask = 1 << sq
        if p1_bb & mask:
            p1.append(sq)
        if p1_ball_bb & mask:
            p1_ball = sq
        if p2_bb & mask:
            p2.append(sq)
        if p2_ball_bb & mask:
            p2_ball = sq
    return p1, p1_ball, p2, p2_ball


def get_square_color(row, col, is_goal_p1, is_goal_p2, is_highlight):
    """Get the fill color for a square, accounting for highlights and goal rows."""
    light = (row + col) % 2 == 0

    if is_highlight:
        return HIGHLIGHT_LIGHT if light else HIGHLIGHT_DARK
    elif is_goal_p1:  # Row 7 (top) — red's home / blue's goal
        return GOAL_P1_LIGHT if light else GOAL_P1_DARK
    elif is_goal_p2:  # Row 0 (bottom) — blue's home / red's goal
        return GOAL_P2_LIGHT if light else GOAL_P2_DARK
    else:
        return LIGHT_SQUARE if light else DARK_SQUARE


def draw_board(state, highlights=None, move_label=None, winner=None, is_final=False):
    """Render a board position as a PIL Image (no alpha compositing)."""
    width = BOARD_MARGIN * 2 + LABEL_SPACE + COLS * CELL_SIZE
    height = BOARD_MARGIN * 2 + ROWS * CELL_SIZE + LABEL_SPACE + 32

    # Use RGB directly — no alpha compositing that washes things out
    img = Image.new('RGB', (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = font
        bold_font = font

    highlight_set = set(highlights) if highlights else set()

    # Draw squares
    for row in range(ROWS):
        for col in range(COLS):
            vr = ROWS - 1 - row
            x0 = BOARD_MARGIN + LABEL_SPACE + col * CELL_SIZE
            y0 = BOARD_MARGIN + vr * CELL_SIZE
            sq = row * COLS + col

            color = get_square_color(
                row, col,
                is_goal_p1=(row == ROWS - 1),
                is_goal_p2=(row == 0),
                is_highlight=(sq in highlight_set),
            )
            draw.rectangle([x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE], fill=color)

    # Labels
    for row in range(ROWS):
        vr = ROWS - 1 - row
        y = BOARD_MARGIN + vr * CELL_SIZE + CELL_SIZE // 2 - 6
        draw.text((BOARD_MARGIN + 5, y), ROW_LABELS[row], fill=TEXT_COLOR, font=small_font)
    for col in range(COLS):
        x = BOARD_MARGIN + LABEL_SPACE + col * CELL_SIZE + CELL_SIZE // 2 - 4
        draw.text((x, BOARD_MARGIN + ROWS * CELL_SIZE + 4), COL_LABELS[col], fill=TEXT_COLOR, font=small_font)

    # Pieces
    p1_pieces, p1_ball, p2_pieces, p2_ball = get_pieces(state)

    def draw_diamond(sq, fill, outline, has_ball):
        cx, cy = rc_to_pixel(*square_to_rc(sq))
        r = PIECE_RADIUS
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(pts, fill=fill, outline=outline, width=2)
        if has_ball:
            br = BALL_RADIUS
            draw.ellipse([cx - br, cy - br, cx + br, cy + br],
                        fill=BALL_COLOR, outline=BALL_OUTLINE, width=2)

    for sq in p1_pieces:
        draw_diamond(sq, P1_COLOR, P1_OUTLINE, sq == p1_ball)
    for sq in p2_pieces:
        draw_diamond(sq, P2_COLOR, P2_OUTLINE, sq == p2_ball)

    # Status text
    sy = BOARD_MARGIN + ROWS * CELL_SIZE + LABEL_SPACE + 10
    if is_final and winner is not None:
        txt = "Blue wins!" if winner == 0 else "Red wins!"
        col = WINNER_BLUE if winner == 0 else WINNER_RED
        draw.text((BOARD_MARGIN + LABEL_SPACE, sy), txt, fill=col, font=bold_font)
    elif move_label:
        draw.text((BOARD_MARGIN + LABEL_SPACE, sy), move_label, fill=TEXT_COLOR, font=small_font)

    return img


def generate_gif(moves, winner, output_path, turn_duration=900, final_hold=3000):
    """Generate animated GIF showing turns (not atomic moves)."""
    frames = []
    durations = []

    state = GameState.new_game()

    # Initial position
    frames.append(draw_board(state, move_label="Starting position"))
    durations.append(2000)

    # Group moves into turns.
    # Knight moves end the turn immediately. Passes chain until END_TURN or game end.
    turns = []
    current_turn = []
    temp_state = GameState.new_game()
    for m in moves:
        if m == -1:
            current_turn.append(m)
            turns.append(current_turn)
            current_turn = []
            temp_state.apply_move(m)
            continue
        src = m // 56
        is_pass = bool(temp_state.my_ball & (1 << src))
        current_turn.append(m)
        temp_state.apply_move(m)
        if not is_pass:
            turns.append(current_turn)
            current_turn = []
    if current_turn:
        turns.append(current_turn)

    turn_num = 0
    for turn_moves in turns:
        highlights = set()
        is_pass_turn = False
        n_actions = 0

        # Track who is moving BEFORE applying moves
        moving_player = state.current_player

        for m in turn_moves:
            if m == -1:
                state.apply_move(m)
                continue
            src = m // 56
            dst = m % 56
            highlights.add(src)
            highlights.add(dst)
            if state.my_ball & (1 << src):
                is_pass_turn = True
            n_actions += 1
            state.apply_move(m)

        turn_num += 1
        player_name = "Blue" if moving_player == 0 else "Red"

        if is_pass_turn and n_actions > 1:
            label = f"Turn {turn_num}: {player_name} passes x{n_actions}"
        elif is_pass_turn:
            label = f"Turn {turn_num}: {player_name} passes"
        else:
            label = f"Turn {turn_num}: {player_name} moves"

        # Check if game is over after this turn
        if state.is_terminal():
            frame = draw_board(state, highlights=highlights, winner=winner, is_final=True)
            frames.append(frame)
            durations.append(final_hold)
        else:
            frame = draw_board(state, highlights=highlights, move_label=label)
            frames.append(frame)
            durations.append(turn_duration + max(0, n_actions - 1) * 300)

    # Convert to palette mode with high-quality dithering for better color
    palette_frames = []
    for f in frames:
        pf = f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        palette_frames.append(pf)

    palette_frames[0].save(
        output_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"Generated {output_path}: {len(frames)} frames, {turn_num} turns")


if __name__ == '__main__':
    import sqlite3
    import json

    db_path = '/app/server/data/games.db'
    game_id = sys.argv[1] if len(sys.argv) > 1 else 'b5b133cc'
    output = sys.argv[2] if len(sys.argv) > 2 else '/tmp/game_replay.gif'

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT moves_json, winner FROM games WHERE game_id LIKE ?", (game_id + '%',))
    row = cur.fetchone()
    if not row:
        print(f"Game {game_id} not found")
        sys.exit(1)

    moves = json.loads(row[0])
    winner = row[1]
    conn.close()

    print(f"Game {game_id}: {len(moves)} moves, winner={'Blue' if winner == 0 else 'Red'}")
    generate_gif(moves, winner, output)

#!/usr/bin/env python3
"""
Terminal-based Razzle Dazzle game client.

Play against the AI or watch AI vs AI games.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import (
    get_legal_moves, move_to_algebraic, algebraic_to_move,
    decode_move, encode_move
)
from razzle.core.bitboard import sq_to_algebraic
from razzle.ai.mcts import MCTS, MCTSConfig, play_move
from razzle.ai.network import RazzleNet, create_network
from razzle.ai.evaluator import BatchedEvaluator, DummyEvaluator


def print_board(state: GameState, highlight_moves: list[int] = None) -> None:
    """Print the board with optional move highlighting.

    Symbols:
        X = Player 1 piece with ball
        x = Player 1 piece without ball (eligible)
        O = Player 2 piece with ball
        o = Player 2 piece without ball (eligible)
        Dim pieces = ineligible receivers (must move before receiving passes)
    """
    # ANSI color codes
    GREEN = '\033[92m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    symbols = {}
    ineligible = set()

    # Track ineligible receivers
    for sq in range(56):
        if state.touched_mask & (1 << sq):
            ineligible.add(sq)

    # Place pieces - check for ball first (uppercase = has ball)
    for sq in range(56):
        bit = 1 << sq
        if state.balls[0] & bit:
            symbols[sq] = 'X'  # P1 piece with ball
        elif state.balls[1] & bit:
            symbols[sq] = 'O'  # P2 piece with ball
        elif state.pieces[0] & bit:
            symbols[sq] = 'x'  # P1 piece
        elif state.pieces[1] & bit:
            symbols[sq] = 'o'  # P2 piece
        else:
            symbols[sq] = '.'

    # Highlight destinations if moves provided
    targets = set()
    if highlight_moves:
        for move in highlight_moves:
            if move >= 0:  # Skip END_TURN_MOVE (-1)
                _, dst = decode_move(move)
                targets.add(dst)

    print()
    print("  +" + "-" * 15 + "+")
    for row in range(7, -1, -1):
        line = f"{row + 1} |"
        for col in range(7):
            sq = row * 7 + col
            sym = symbols.get(sq, '.')
            if sq in targets:
                line += f" {GREEN}{sym}{RESET}"  # Green = move target
            elif sq in ineligible and sym != '.':
                line += f" {DIM}{sym}{RESET}"  # Dim = ineligible receiver
            else:
                line += f" {sym}"
        line += " |"
        print(line)
    print("  +" + "-" * 15 + "+")
    print("    a b c d e f g")
    print(f"  X/O=ball  x/o=piece  {DIM}dim{RESET}=ineligible")
    print()


def parse_user_move(state: GameState, input_str: str) -> int | None:
    """Parse user input into a move."""
    input_str = input_str.strip().lower()

    # Check for special commands
    if input_str in ['q', 'quit', 'exit']:
        return 'quit'
    if input_str in ['h', 'help', '?']:
        return 'help'
    if input_str in ['m', 'moves']:
        return 'show_moves'
    if input_str in ['u', 'undo']:
        return 'undo'

    # Try algebraic notation (e.g., "d1-c3")
    try:
        move = algebraic_to_move(input_str)
        if move in get_legal_moves(state):
            return move
        else:
            print(f"Illegal move: {input_str}")
            return None
    except (ValueError, IndexError):
        print(f"Invalid format: {input_str}. Use notation like 'd1-c3'")
        return None


def show_legal_moves(state: GameState) -> None:
    """Display all legal moves."""
    moves = get_legal_moves(state)
    if not moves:
        print("No legal moves!")
        return

    # Separate passes and knight moves
    passes = []
    knight_moves = []

    ball_sq = None
    for sq in range(56):
        if state.balls[state.current_player] & (1 << sq):
            ball_sq = sq
            break

    for move in moves:
        src, dst = decode_move(move)
        if src == ball_sq:
            passes.append(move)
        else:
            knight_moves.append(move)

    if knight_moves:
        print("Knight moves:", ", ".join(move_to_algebraic(m) for m in knight_moves))
    if passes:
        print("Ball passes:", ", ".join(move_to_algebraic(m) for m in passes))


def play_human_vs_ai(
    network: RazzleNet | None = None,
    human_player: int = 0,
    num_simulations: int = 800,
    device: str = 'cpu'
) -> None:
    """Play a game: human vs AI."""
    state = GameState.new_game()

    if network:
        evaluator = BatchedEvaluator(network, device=device)
    else:
        print("No network loaded. Using random policy.")
        evaluator = DummyEvaluator()

    mcts_config = MCTSConfig(num_simulations=num_simulations, temperature=0.0)

    print("\n=== Razzle Dazzle ===")
    print("You are", "X (player 1)" if human_player == 0 else "O (player 2)")
    print("Commands: move (e.g., 'd1-c3'), 'm' for moves, 'u' undo, 'q' quit")
    print("Goal: Get your ball to the opponent's back row!")

    while not state.is_terminal():
        print_board(state)
        player = state.current_player

        if player == human_player:
            # Human's turn
            print(f"Your turn (Player {player + 1})")

            while True:
                try:
                    user_input = input("> ").strip()
                except EOFError:
                    return

                result = parse_user_move(state, user_input)

                if result == 'quit':
                    print("Thanks for playing!")
                    return
                elif result == 'help':
                    print("Enter moves like 'd1-c3' to move a piece")
                    print("'m' to see legal moves, 'u' to undo, 'q' to quit")
                elif result == 'show_moves':
                    show_legal_moves(state)
                elif result == 'undo':
                    if state.history:
                        state.undo_move()
                        if state.history:  # Undo AI move too
                            state.undo_move()
                        print("Move undone.")
                    else:
                        print("Nothing to undo.")
                elif result is not None:
                    state.apply_move(result)
                    print(f"You played: {move_to_algebraic(result)}")
                    break
        else:
            # AI's turn
            print(f"AI thinking ({num_simulations} simulations)...")
            mcts = MCTS(evaluator, mcts_config)
            root = mcts.search(state, add_noise=False)
            move = mcts.select_move(root)

            # Show analysis
            analysis = mcts.analyze(root, top_k=3)
            print("AI analysis:")
            for m in analysis:
                print(f"  {m['algebraic']}: {m['visits']} visits, "
                      f"value={m['value']:.2f}")

            state.apply_move(move)
            print(f"AI plays: {move_to_algebraic(move)}")

    # Game over
    print_board(state)
    winner = state.get_winner()
    if winner is not None:
        if winner == human_player:
            print("Congratulations! You win!")
        else:
            print("AI wins. Better luck next time!")
    else:
        print("Game drawn.")


def watch_ai_vs_ai(
    network: RazzleNet | None = None,
    num_simulations: int = 400,
    device: str = 'cpu',
    delay: float = 1.0
) -> None:
    """Watch AI play against itself."""
    import time

    state = GameState.new_game()

    if network:
        evaluator = BatchedEvaluator(network, device=device)
    else:
        evaluator = DummyEvaluator()

    mcts_config = MCTSConfig(num_simulations=num_simulations, temperature=0.1)

    print("\n=== AI vs AI ===")
    print(f"Simulations per move: {num_simulations}")

    move_count = 0
    while not state.is_terminal():
        print_board(state)
        print(f"Move {move_count + 1}, Player {state.current_player + 1}")

        mcts = MCTS(evaluator, mcts_config)
        root = mcts.search(state)
        move = mcts.select_move(root)

        analysis = mcts.analyze(root, top_k=3)
        for m in analysis:
            print(f"  {m['algebraic']}: {m['visits']} visits, value={m['value']:.2f}")

        state.apply_move(move)
        print(f"Plays: {move_to_algebraic(move)}\n")

        move_count += 1
        time.sleep(delay)

    print_board(state)
    winner = state.get_winner()
    print(f"Game over after {move_count} moves. Winner: Player {winner + 1 if winner is not None else 'None (draw)'}")


def main():
    parser = argparse.ArgumentParser(description='Razzle Dazzle Terminal Client')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--simulations', type=int, default=800, help='MCTS simulations')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--watch', action='store_true', help='Watch AI vs AI')
    parser.add_argument('--play-as', type=int, choices=[1, 2], default=1,
                        help='Play as player 1 (X) or 2 (O)')

    args = parser.parse_args()

    # Load network if provided
    network = None
    if args.model:
        print(f"Loading model from {args.model}...")
        network = RazzleNet.load(args.model, device=args.device)
        print(f"Model loaded ({network.num_parameters()} parameters)")

    if args.watch:
        watch_ai_vs_ai(network, args.simulations, args.device)
    else:
        play_human_vs_ai(network, args.play_as - 1, args.simulations, args.device)


if __name__ == '__main__':
    main()

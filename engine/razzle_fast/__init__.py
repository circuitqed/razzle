"""
razzle_fast - C-accelerated game state and MCTS for Razzle Dazzle.

Provides FastState and FastMCTS as drop-in replacements for the Python
engine, with ~5x speedup on MCTS tree operations.

Build:
    cd engine && python3 razzle_fast/_build.py

Usage:
    from razzle_fast import FastState, FastMCTS
"""

try:
    from .wrapper import FastState, FastMCTS
except (ImportError, OSError) as e:
    import sys
    print(
        f"razzle_fast: C library not built. Run: cd engine && python3 razzle_fast/_build.py\n"
        f"Error: {e}",
        file=sys.stderr,
    )
    raise

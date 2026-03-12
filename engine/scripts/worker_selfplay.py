#!/usr/bin/env python3
"""
Distributed self-play worker for Razzle Dazzle training.

This script runs on cloud GPU instances (Vast.ai) and:
1. Plays games continuously using MCTS + neural network
2. Submits completed games to the training API
3. Writes status updates to status.json for monitoring
4. Periodically checks for new model weights via API and hot-reloads

Usage:
    python worker_selfplay.py --worker-id 0 --api-url http://server:8000 --simulations 400

Status file (status.json):
    {
        "worker_id": 0,
        "status": "running",
        "games_completed": 42,
        "current_game_moves": 73,
        "model_version": "iter_002",
        "last_update": "2024-01-20T15:30:45",
        "uptime_sec": 3600,
        "games_per_hour": 12.5
    }
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.ai.network import RazzleNet, create_network
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import BatchedEvaluator
from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.training.api_client import TrainingAPIClient
import math

# Try to use C-backed FastMCTS for much faster tree search
try:
    from razzle_fast.wrapper import FastMCTS
    HAS_FAST_MCTS = True
except ImportError:
    HAS_FAST_MCTS = False


@dataclass
class WorkerStatus:
    """Status information for monitoring."""
    worker_id: str
    status: str  # starting, running, paused, stopped, error
    games_completed: int
    games_submitted: int
    current_game_moves: int
    model_version: str
    last_update: str
    start_time: str
    uptime_sec: float
    games_per_hour: float
    error_message: Optional[str] = None
    gpu_name: str = ""
    simulations: int = 0
    api_url: str = ""


class SelfPlayWorker:
    """
    Continuous self-play worker that submits games via API.

    Generates games independently and submits them to the training server.
    """

    def __init__(
        self,
        worker_id: str,
        api_url: str,
        workspace: Path,
        device: str = 'cuda',
        simulations: int = 400,
        temperature_moves: int = 30,
        network_size: str = 'medium',
        filters: int = 0,
        blocks: int = 0,
        batch_size: int = 32,
        model_check_interval: int = 5,  # Check for new model every N games
        random_opening_moves: int = 0,  # Number of random moves at start
        random_opening_fraction: float = 0.0,  # Fraction of games with random opening
        arena_fraction: float = 0.1,  # Fraction of games that are arena matches
        max_games: int = 0,  # Stop after N games (0 = unlimited)
    ):
        self.worker_id = worker_id
        self.api_url = api_url
        self.workspace = Path(workspace)
        self.device = device
        self.simulations = simulations
        self.temperature_moves = temperature_moves
        self.network_size = network_size
        self.filters = filters
        self.blocks = blocks
        self.batch_size = batch_size
        self.model_check_interval = model_check_interval
        self.random_opening_moves = random_opening_moves
        self.random_opening_fraction = random_opening_fraction
        self.arena_fraction = arena_fraction
        self.max_games = max_games

        # Directories
        self.model_dir = self.workspace / "model"
        self.status_file = self.workspace / "status.json"

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # API client
        self.api_client = TrainingAPIClient(base_url=api_url)

        # State
        self.network: Optional[RazzleNet] = None
        self.evaluator = None
        self.model_version = "none"
        self.games_completed = 0
        self.games_submitted = 0
        self.start_time = datetime.now()
        self.current_game_moves = 0
        self.shutdown_event = Event()
        self.status = "starting"
        self.error_message = None

        # GPU info
        self.gpu_name = self._get_gpu_name()

        # Status writer thread
        self._status_thread: Optional[Thread] = None

        # Arena tracking
        self.available_models: list[str] = []  # List of model versions
        self.arena_games_played = 0
        self.model_cache: dict[str, RazzleNet] = {}  # Cache loaded models

    def _get_gpu_name(self) -> str:
        """Get GPU name if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except:
            pass
        return "unknown"

    def _load_model_to_device(self, network: RazzleNet) -> BatchedEvaluator:
        """Move network to device and create evaluator.

        Loads on CPU first, then moves to GPU to avoid CUDA initialization
        races when multiple workers share a GPU.
        """
        return BatchedEvaluator(
            network,
            batch_size=self.batch_size,
            device=self.device
        )

    def _download_latest_model(self) -> bool:
        """Download the latest model from the API."""
        try:
            model_info = self.api_client.get_latest_model()
            if model_info is None:
                return False

            if model_info.version == self.model_version:
                return False  # Already have this version

            # Download model
            model_path = self.model_dir / f"{model_info.version}.pt"
            print(f"[Worker {self.worker_id}] Downloading model: {model_info.version}")
            self.api_client.download_model(model_info.version, model_path)

            # Load model on CPU first, then move to device
            self.network = RazzleNet.load(model_path, device='cpu')
            self.evaluator = self._load_model_to_device(self.network)
            self.model_version = model_info.version
            print(f"[Worker {self.worker_id}] Loaded model: {self.model_version}")
            return True

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error downloading model: {e}")
            return False

    def _load_or_create_network(self) -> bool:
        """Load model from API or create new one.

        Retries on failure (e.g. transient CUDA errors when multiple
        workers initialize on the same GPU).
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # First try to download from API
                if self._download_latest_model():
                    return True

                # Check local model directory
                model_files = sorted(self.model_dir.glob("*.pt"))
                if model_files:
                    latest = model_files[-1]
                    self.network = RazzleNet.load(latest, device='cpu')
                    self.model_version = latest.stem
                    print(f"[Worker {self.worker_id}] Loaded local model: {self.model_version}")
                else:
                    # Create new network (on CPU, evaluator moves to device)
                    if self.filters > 0 and self.blocks > 0:
                        self.network = create_network(num_filters=self.filters, num_blocks=self.blocks, device='cpu')
                    else:
                        self.network = create_network(preset=self.network_size, device='cpu')
                    self.model_version = "initial"
                    print(f"[Worker {self.worker_id}] Created new network")

                # Create evaluator (moves model to device)
                self.evaluator = self._load_model_to_device(self.network)
                return True

            except Exception as e:
                self.error_message = f"Failed to load network: {e}"
                print(f"[Worker {self.worker_id}] {self.error_message}")
                if attempt < max_retries - 1:
                    delay = 10 * (attempt + 1)
                    print(f"[Worker {self.worker_id}] Retrying in {delay}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(delay)

        return False

    def _check_for_new_model(self) -> bool:
        """Check if a new model is available via API."""
        try:
            model_info = self.api_client.get_latest_model()
            if model_info is None:
                return False

            if model_info.version != self.model_version:
                return self._download_latest_model()

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error checking for new model: {e}")

        return False

    def _refresh_available_models(self) -> list[str]:
        """Get benchmark models (every 10th iteration + latest) for arena matches."""
        try:
            import requests
            resp = requests.get(f"{self.api_url}/training/models", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get('models', [])

                # Parse iteration numbers
                parsed = []
                for m in models:
                    v = m.get('version', '')
                    if not v:
                        continue
                    if v == 'initial':
                        parsed.append((0, v))
                    else:
                        try:
                            parsed.append((int(v.split('_')[1]), v))
                        except (IndexError, ValueError):
                            continue

                if not parsed:
                    return self.available_models

                parsed.sort(key=lambda x: x[0])
                max_iter = parsed[-1][0]

                # Keep only benchmark iterations (mod 10) + the latest
                benchmarks = []
                for iteration, version in parsed:
                    if iteration % 10 == 0 or iteration == max_iter:
                        benchmarks.append(version)

                self.available_models = benchmarks
                return benchmarks
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error fetching model list: {e}")
        return self.available_models

    def _get_model_elos(self) -> dict[str, float]:
        """Fetch Elo ratings for models from API.

        Filters to the highest-sim-count rating per model to get
        the most reliable estimates.
        """
        try:
            import requests
            resp = requests.get(f"{self.api_url}/arena/ratings", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                ratings = data.get('ratings', [])
                # Keep highest-sim-count entry per model for reliability
                elos: dict[str, tuple[int, float]] = {}  # version -> (sims, elo)
                for r in ratings:
                    version = r.get('model_version', '')
                    elo = r.get('elo_rating', 1200)
                    sims = r.get('simulations', 0)
                    games = r.get('games_played', 0)
                    if not version or games < 10:
                        continue
                    if version not in elos or sims > elos[version][0]:
                        elos[version] = (sims, elo)
                return {v: elo for v, (_, elo) in elos.items()}
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error fetching Elo ratings: {e}")
        return {}

    @staticmethod
    def _iter_number(version: str) -> int:
        """Extract iteration number from version string."""
        if version == 'initial':
            return 0
        try:
            return int(version.split('_')[1])
        except (IndexError, ValueError):
            return 0

    def _interpolate_elo(self, version: str, elos: dict[str, float], all_models: list[str]) -> float:
        """Estimate Elo for an unrated model by interpolating from neighbors."""
        target_iter = self._iter_number(version)

        # Find nearest rated neighbors
        below, above = None, None
        for m in all_models:
            if m not in elos:
                continue
            it = self._iter_number(m)
            if it <= target_iter and (below is None or it > self._iter_number(below)):
                below = m
            if it >= target_iter and (above is None or it < self._iter_number(above)):
                above = m

        if below and above and below != above:
            # Linear interpolation
            it_lo, it_hi = self._iter_number(below), self._iter_number(above)
            t = (target_iter - it_lo) / (it_hi - it_lo) if it_hi != it_lo else 0.5
            return elos[below] + t * (elos[above] - elos[below])
        elif below:
            return elos[below]
        elif above:
            return elos[above]
        return 1200.0  # fallback

    def _select_arena_models(self) -> tuple[str, str]:
        """
        Select two models for arena match using entropy-weighted pair sampling.

        Restricts pool to benchmark iterations (mod 10) + latest model.
        Weights each pair by softmax(entropy(expected_outcome) / temperature),
        so competitive matchups (close Elo) are strongly preferred while
        still allowing occasional wider-gap games.
        """
        SOFTMAX_TEMP = 0.25

        models = self._refresh_available_models()
        if len(models) < 2:
            return (self.model_version, self.model_version)

        # Get Elo ratings, interpolate for unrated models
        elos = self._get_model_elos()
        model_elos = {}
        for m in models:
            if m in elos:
                model_elos[m] = elos[m]
            else:
                model_elos[m] = self._interpolate_elo(m, elos, models)

        # Build all pairs and compute entropy weights
        from itertools import combinations
        import math

        pairs = list(combinations(range(len(models)), 2))
        if not pairs:
            return (self.model_version, self.model_version)

        raw_entropies = np.zeros(len(pairs))
        for k, (i, j) in enumerate(pairs):
            diff = abs(model_elos[models[i]] - model_elos[models[j]])
            win_prob = 1.0 / (1.0 + 10 ** (-diff / 400))
            # Binary entropy
            if win_prob <= 0.001 or win_prob >= 0.999:
                raw_entropies[k] = 0.001
            else:
                raw_entropies[k] = -(win_prob * math.log2(win_prob)
                                     + (1 - win_prob) * math.log2(1 - win_prob))

        # Softmax with temperature to sharpen distribution toward competitive matchups
        logits = raw_entropies / SOFTMAX_TEMP
        logits -= logits.max()  # numerical stability
        exp_w = np.exp(logits)
        probs = exp_w / exp_w.sum()

        # Sample a pair
        idx = np.random.choice(len(pairs), p=probs)
        i, j = pairs[idx]

        return (models[i], models[j])

    def _get_model_evaluator(self, version: str) -> BatchedEvaluator:
        """Get evaluator for a specific model version, using cache."""
        if version == self.model_version:
            return self.evaluator

        if version in self.model_cache:
            net = self.model_cache[version]
        else:
            # Download and load model
            model_path = self.model_dir / f"{version}.pt"
            if not model_path.exists():
                print(f"[Worker {self.worker_id}] Downloading model for arena: {version}")
                self.api_client.download_model(version, model_path)

            net = RazzleNet.load(model_path, device='cpu')
            # Keep cache small (max 5 models)
            if len(self.model_cache) >= 5:
                # Remove oldest
                oldest = next(iter(self.model_cache))
                del self.model_cache[oldest]
            self.model_cache[version] = net

        return self._load_model_to_device(net)

    def play_arena_game(self, model1_version: str, model2_version: str) -> tuple[list[int], float, list[dict[int, int]], str, str]:
        """
        Play an arena game between two models.

        Returns:
            Tuple of (moves, result, visit_counts, model1_version, model2_version)
            result is from model1's perspective: +1 if model1 won, -1 if model2 won, 0 draw
        """
        eval1 = self._get_model_evaluator(model1_version)
        eval2 = self._get_model_evaluator(model2_version)

        state = GameState.new_game()
        moves = []
        visit_counts = []

        move_count = 0
        self.current_game_moves = 0

        # Track which evaluator plays which side
        # model1 plays as player 0, model2 plays as player 1
        evaluators = [eval1, eval2]

        while not state.is_terminal() and move_count < 300:
            current_player = state.current_player
            ev = evaluators[current_player]

            # Use temperature for opening diversity (first 3 moves)
            temp = 1.0 if move_count < 6 else 0.0
            # Mid-pass positions have narrow trees — use fewer sims
            sims = self.simulations
            if state.has_passed:
                sims = max(50, sims // 10)
            config = MCTSConfig(
                num_simulations=sims,
                temperature=temp,
                batch_size=self.batch_size
            )

            if HAS_FAST_MCTS:
                fast_mcts = FastMCTS(ev, config)
                result_obj, vc = fast_mcts.search_with_visits(state, add_noise=(move_count < 6))
                visit_counts.append(vc)
                move = result_obj.selected_move
            else:
                mcts = MCTS(ev, config)
                root = mcts.search_batched(state, add_noise=(move_count < 6))
                vc = {}
                for m, child in root.children.items():
                    if child.visit_count > 0:
                        vc[m] = child.visit_count
                visit_counts.append(vc)
                move = mcts.select_move(root)
            moves.append(move)
            state.apply_move(move)

            move_count += 1
            self.current_game_moves = move_count

            if self.shutdown_event.is_set():
                break

        # Determine result from model1's perspective
        winner = state.get_winner()
        if winner == 0:
            result = 1.0  # model1 won
        elif winner == 1:
            result = -1.0  # model2 won
        else:
            result = 0.0  # draw

        return moves, result, visit_counts, model1_version, model2_version

    def submit_arena_result(self, model1: str, model2: str, result: float) -> bool:
        """Submit arena match result to API."""
        try:
            import requests
            # result is from model1's perspective
            wins1 = 1 if result > 0 else 0
            wins2 = 1 if result < 0 else 0
            draws = 1 if result == 0 else 0

            data = {
                'model1_version': model1,
                'model2_version': model2,
                'model1_wins': wins1,
                'model2_wins': wins2,
                'draws': draws,
                'simulations': self.simulations,
            }
            headers = {}
            api_key = os.environ.get("TRAINING_API_KEY", "")
            if api_key:
                headers["X-API-Key"] = api_key
            resp = requests.post(f"{self.api_url}/arena/matches", json=data, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"[Worker {self.worker_id}] Arena submit failed: {resp.status_code}")
            return resp.status_code == 200
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error submitting arena result: {e}")
            return False

    def _get_status(self) -> WorkerStatus:
        """Get current worker status."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        games_per_hour = (self.games_completed / uptime * 3600) if uptime > 0 else 0.0

        return WorkerStatus(
            worker_id=self.worker_id,
            status=self.status,
            games_completed=self.games_completed,
            games_submitted=self.games_submitted,
            current_game_moves=self.current_game_moves,
            model_version=self.model_version,
            last_update=datetime.now().isoformat(),
            start_time=self.start_time.isoformat(),
            uptime_sec=uptime,
            games_per_hour=games_per_hour,
            error_message=self.error_message,
            gpu_name=self.gpu_name,
            simulations=self.simulations,
            api_url=self.api_url,
        )

    def _write_status(self):
        """Write status to file."""
        status = self._get_status()

        # Write atomically
        tmp_path = self.status_file.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(asdict(status), f, indent=2)
        tmp_path.rename(self.status_file)

    def _status_writer_loop(self):
        """Background thread to periodically write status."""
        while not self.shutdown_event.is_set():
            try:
                self._write_status()
            except Exception as e:
                print(f"[Worker {self.worker_id}] Error writing status: {e}")

            # Wait 5 seconds or until shutdown
            self.shutdown_event.wait(5.0)

    def _get_ball_row(self, state: GameState, player: int) -> int:
        """Get the row (0-7) of the specified player's ball."""
        for sq in range(56):
            if state.balls[player] & (1 << sq):
                return sq // 7
        return 0

    def play_one_game(self) -> tuple[list[int], float, list[dict[int, int]]]:
        """
        Play a single self-play game.

        If random_opening_fraction > 0, some games use random moves for
        the first random_opening_moves moves to increase opening diversity.

        Returns:
            Tuple of (moves, result, visit_counts)
        """
        state = GameState.new_game()
        moves = []
        visit_counts = []

        move_count = 0
        self.current_game_moves = 0

        # Decide if this game uses random opening
        use_random_opening = random.random() < self.random_opening_fraction

        while not state.is_terminal() and move_count < 300:
            legal_moves = get_legal_moves(state)

            # Check if we should use random move (for opening diversity)
            if use_random_opening and move_count < self.random_opening_moves:
                # Random opening move - uniform distribution over legal moves
                move = random.choice(legal_moves)

                # Record uniform visit counts for training
                vc = {m: 1 for m in legal_moves}
                visit_counts.append(vc)

                moves.append(move)
                state.apply_move(move)
            else:
                # Standard MCTS move
                temp = 1.0 if move_count < self.temperature_moves else 0.0
                # Mid-pass positions have narrow trees — use fewer sims
                sims = self.simulations
                if state.has_passed:
                    sims = max(50, sims // 10)
                config = MCTSConfig(
                    num_simulations=sims,
                    temperature=temp,
                    batch_size=self.batch_size
                )

                if HAS_FAST_MCTS:
                    fast_mcts = FastMCTS(self.evaluator, config)
                    result_obj, vc = fast_mcts.search_with_visits(state, add_noise=True)
                    visit_counts.append(vc)
                    move = result_obj.selected_move
                else:
                    mcts = MCTS(self.evaluator, config)
                    root = mcts.search_batched(state, add_noise=True)
                    vc = {}
                    for m, child in root.children.items():
                        if child.visit_count > 0:
                            vc[m] = child.visit_count
                    visit_counts.append(vc)
                    move = mcts.select_move(root)

                moves.append(move)
                state.apply_move(move)

            move_count += 1
            self.current_game_moves = move_count

            # Check for shutdown
            if self.shutdown_event.is_set():
                break

        # Determine result
        winner = state.get_winner()
        if winner == 0:
            result = 1.0
        elif winner == 1:
            result = -1.0
        else:
            result = 0.0

        return moves, result, visit_counts

    def submit_game(self, moves: list[int], result: float, visit_counts: list[dict[int, int]]) -> bool:
        """Submit a game to the API."""
        try:
            game_id = self.api_client.submit_game(
                worker_id=self.worker_id,
                moves=moves,
                result=result,
                visit_counts=visit_counts,
                model_version=self.model_version,
            )
            self.games_submitted += 1
            return True
        except Exception as e:
            print(f"[Worker {self.worker_id}] Failed to submit game: {e}")
            return False

    def run(self):
        """Main worker loop."""
        print(f"[Worker {self.worker_id}] Starting on {self.device}")
        print(f"[Worker {self.worker_id}] API URL: {self.api_url}")
        print(f"[Worker {self.worker_id}] Simulations: {self.simulations}")

        # Wait for API to be available
        print(f"[Worker {self.worker_id}] Waiting for API server...")
        if not self.api_client.wait_for_server(timeout=120):
            self.status = "error"
            self.error_message = "API server not available"
            print(f"[Worker {self.worker_id}] {self.error_message}")
            self._write_status()
            return

        print(f"[Worker {self.worker_id}] API server connected")

        # Load network
        if not self._load_or_create_network():
            self.status = "error"
            self._write_status()
            return

        self.status = "running"

        # Start status writer thread
        self._status_thread = Thread(target=self._status_writer_loop, daemon=True)
        self._status_thread.start()

        mcts_type = "FastMCTS (C)" if HAS_FAST_MCTS else "MCTS (Python)"
        print(f"[Worker {self.worker_id}] Starting self-play loop (arena fraction: {self.arena_fraction:.0%}, engine: {mcts_type})")

        try:
            while not self.shutdown_event.is_set():
                # Decide if this is an arena game
                is_arena_game = random.random() < self.arena_fraction

                if is_arena_game and len(self.available_models) >= 2:
                    # Play arena game between two different models
                    model1, model2 = self._select_arena_models()
                    if model1 != model2:
                        moves, result, visit_counts, m1, m2 = self.play_arena_game(model1, model2)

                        if self.shutdown_event.is_set():
                            break

                        # Submit as training game (use model1's version for training data)
                        if self.submit_game(moves, result, visit_counts):
                            self.games_completed += 1
                            self.arena_games_played += 1

                            # Also submit arena result
                            self.submit_arena_result(m1, m2, result)

                            winner_str = {1.0: m1, -1.0: m2, 0.0: "Draw"}.get(result, "?")
                            print(f"[Worker {self.worker_id}] Arena {self.arena_games_played}: "
                                  f"{m1} vs {m2}, {len(moves)} moves, winner={winner_str}")
                    else:
                        is_arena_game = False  # Fall back to self-play

                if not is_arena_game:
                    # Standard self-play game
                    moves, result, visit_counts = self.play_one_game()

                    if self.shutdown_event.is_set():
                        break

                    # Submit game to API
                    if self.submit_game(moves, result, visit_counts):
                        self.games_completed += 1

                        # Log progress
                        winner_str = {1.0: "P1", -1.0: "P2", 0.0: "Draw"}.get(result, "?")
                        print(f"[Worker {self.worker_id}] Game {self.games_completed}: "
                              f"{len(moves)} moves, winner={winner_str}")

                # Check game limit
                if self.max_games > 0 and self.games_completed >= self.max_games:
                    print(f"[Worker {self.worker_id}] Reached game limit ({self.max_games})")
                    break

                # Check for new model periodically
                if self.games_completed % self.model_check_interval == 0:
                    self._check_for_new_model()
                    # Also refresh model list for arena
                    if self.arena_fraction > 0:
                        self._refresh_available_models()

        except Exception as e:
            self.status = "error"
            self.error_message = str(e)
            print(f"[Worker {self.worker_id}] Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.status = "stopped"
            self._write_status()
            print(f"[Worker {self.worker_id}] Stopped. Total games: {self.games_completed}")

    def shutdown(self):
        """Signal the worker to stop."""
        print(f"[Worker {self.worker_id}] Shutdown requested")
        self.shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(description='Distributed self-play worker')
    parser.add_argument('--worker-id', type=str, required=True, help='Unique worker ID (e.g. a1b2c3_0)')
    parser.add_argument('--api-url', type=str, required=True,
                        help='Training API URL (e.g., http://server:8000)')
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'),
                        help='Workspace directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move')
    parser.add_argument('--temperature-moves', type=int, default=30,
                        help='Number of moves to use temperature')
    parser.add_argument('--filters', type=int, default=None,
                        help='Network filter count (overrides --network-size)')
    parser.add_argument('--blocks', type=int, default=None,
                        help='Network residual blocks (overrides --network-size)')
    parser.add_argument('--network-size', type=str, default='medium', choices=['small', 'medium', 'large'],
                        help='Network size preset: small (~236K), medium (~2.4M), large (~24M)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='MCTS batch size for GPU parallelism')
    parser.add_argument('--model-check-interval', type=int, default=5,
                        help='Check for new model every N games')
    parser.add_argument('--random-opening-moves', type=int, default=0,
                        help='Number of random moves at game start (default: 0)')
    parser.add_argument('--random-opening-fraction', type=float, default=0.3,
                        help='Fraction of games with random openings (default: 0.3)')
    parser.add_argument('--arena-fraction', type=float, default=0.1,
                        help='Fraction of games that are arena matches between models (default: 0.1)')
    parser.add_argument('--max-games', type=int, default=0,
                        help='Stop after N games (0 = unlimited)')

    args = parser.parse_args()

    # Resolve network size
    network_size = args.network_size
    filters = args.filters or 0
    blocks = args.blocks or 0

    # Ensure unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'

    worker = SelfPlayWorker(
        worker_id=args.worker_id,
        api_url=args.api_url,
        workspace=args.workspace,
        device=args.device,
        simulations=args.simulations,
        temperature_moves=args.temperature_moves,
        network_size=network_size,
        filters=filters,
        blocks=blocks,
        batch_size=args.batch_size,
        model_check_interval=args.model_check_interval,
        random_opening_moves=args.random_opening_moves,
        random_opening_fraction=args.random_opening_fraction,
        arena_fraction=args.arena_fraction,
        max_games=args.max_games,
    )

    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        worker.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker.run()


if __name__ == '__main__':
    main()

"""
Permanent storage for training games.

Games are stored as compressed numpy arrays for efficient loading during training.
This is separate from the training pipeline database to avoid bloat and allow
easy reuse of games across different training runs.

Storage format:
    games_archive/
    ├── manifest.json           # Index of all archived batches
    ├── iter_001_batch_001.npz  # Compressed training examples
    ├── iter_001_batch_002.npz
    └── ...

Each .npz file contains:
    - states: (N, 7, 8, 7) float32 - board state tensors
    - policies: (N, 3137) float32 - MCTS policy targets
    - values: (N,) float32 - game outcome from player's perspective
    - legal_masks: (N, 3137) float32 - legal move masks
    - metadata: JSON string with batch info
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
import threading


@dataclass
class BatchMetadata:
    """Metadata for a batch of archived games."""
    batch_id: str
    model_version: str
    num_games: int
    num_examples: int
    avg_game_length: float
    p0_wins: int
    p1_wins: int
    draws: int
    simulations: int
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> 'BatchMetadata':
        return cls(**json.loads(s))


class GameArchive:
    """
    Manages permanent storage of training games.

    Thread-safe for concurrent writes from multiple workers.
    """

    def __init__(self, archive_dir: str = "games_archive"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.archive_dir / "manifest.json"
        self._lock = threading.Lock()
        self._load_manifest()

    def _load_manifest(self):
        """Load or create the manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {
                "version": 1,
                "created_at": datetime.utcnow().isoformat(),
                "batches": {},
                "stats": {
                    "total_games": 0,
                    "total_examples": 0,
                }
            }
            self._save_manifest()

    def _save_manifest(self):
        """Save the manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def _generate_batch_id(self, model_version: str) -> str:
        """Generate a unique batch ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        # Count existing batches for this model version
        existing = [b for b in self.manifest["batches"] if b.startswith(model_version)]
        batch_num = len(existing) + 1
        return f"{model_version}_batch_{batch_num:04d}_{timestamp}"

    def archive_games(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        legal_masks: np.ndarray,
        model_version: str,
        num_games: int,
        game_results: list[float],
        simulations: int = 800,
    ) -> str:
        """
        Archive a batch of training examples.

        Args:
            states: Board state tensors (N, 7, 8, 7)
            policies: MCTS policy targets (N, 3137)
            values: Value targets (N,)
            legal_masks: Legal move masks (N, 3137)
            model_version: Model that generated these games
            num_games: Number of games in this batch
            game_results: List of game results (1.0 = P0 win, -1.0 = P1 win, 0 = draw)
            simulations: MCTS simulations per move

        Returns:
            batch_id: Unique identifier for this batch
        """
        with self._lock:
            batch_id = self._generate_batch_id(model_version)

            # Calculate stats
            p0_wins = sum(1 for r in game_results if r > 0)
            p1_wins = sum(1 for r in game_results if r < 0)
            draws = sum(1 for r in game_results if r == 0)
            avg_length = len(states) / max(num_games, 1)

            # Create metadata
            metadata = BatchMetadata(
                batch_id=batch_id,
                model_version=model_version,
                num_games=num_games,
                num_examples=len(states),
                avg_game_length=avg_length,
                p0_wins=p0_wins,
                p1_wins=p1_wins,
                draws=draws,
                simulations=simulations,
                created_at=datetime.utcnow().isoformat(),
            )

            # Save to compressed numpy file
            batch_path = self.archive_dir / f"{batch_id}.npz"
            np.savez_compressed(
                batch_path,
                states=states.astype(np.float32),
                policies=policies.astype(np.float32),
                values=values.astype(np.float32),
                legal_masks=legal_masks.astype(np.float32),
                metadata=metadata.to_json(),
            )

            # Update manifest
            self.manifest["batches"][batch_id] = {
                "file": f"{batch_id}.npz",
                "model_version": model_version,
                "num_games": num_games,
                "num_examples": len(states),
                "created_at": metadata.created_at,
            }
            self.manifest["stats"]["total_games"] += num_games
            self.manifest["stats"]["total_examples"] += len(states)
            self._save_manifest()

            return batch_id

    def load_batch(self, batch_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, BatchMetadata]:
        """
        Load a batch of training examples.

        Returns:
            (states, policies, values, legal_masks, metadata)
        """
        batch_path = self.archive_dir / f"{batch_id}.npz"
        data = np.load(batch_path, allow_pickle=True)

        metadata = BatchMetadata.from_json(str(data["metadata"]))

        return (
            data["states"],
            data["policies"],
            data["values"],
            data["legal_masks"],
            metadata,
        )

    def load_batches(
        self,
        model_versions: Optional[list[str]] = None,
        min_examples: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load multiple batches, optionally filtering by model version.

        Args:
            model_versions: Only load batches from these models (None = all)
            min_examples: Skip batches with fewer examples

        Returns:
            Combined (states, policies, values, legal_masks)
        """
        all_states = []
        all_policies = []
        all_values = []
        all_masks = []

        for batch_id, info in self.manifest["batches"].items():
            # Filter by model version
            if model_versions and info["model_version"] not in model_versions:
                continue

            # Filter by size
            if info["num_examples"] < min_examples:
                continue

            states, policies, values, masks, _ = self.load_batch(batch_id)
            all_states.append(states)
            all_policies.append(policies)
            all_values.append(values)
            all_masks.append(masks)

        if not all_states:
            return (
                np.zeros((0, 7, 8, 7), dtype=np.float32),
                np.zeros((0, 3137), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 3137), dtype=np.float32),
            )

        return (
            np.concatenate(all_states),
            np.concatenate(all_policies),
            np.concatenate(all_values),
            np.concatenate(all_masks),
        )

    def get_stats(self) -> dict:
        """Get archive statistics."""
        return {
            "total_batches": len(self.manifest["batches"]),
            "total_games": self.manifest["stats"]["total_games"],
            "total_examples": self.manifest["stats"]["total_examples"],
            "models": list(set(b["model_version"] for b in self.manifest["batches"].values())),
        }

    def list_batches(self, model_version: Optional[str] = None) -> list[dict]:
        """List all batches, optionally filtered by model version."""
        batches = []
        for batch_id, info in self.manifest["batches"].items():
            if model_version and info["model_version"] != model_version:
                continue
            batches.append({"batch_id": batch_id, **info})
        return sorted(batches, key=lambda x: x["created_at"])


def archive_from_game_records(
    records: list,  # List of GameRecord objects
    archive: GameArchive,
    model_version: str,
    simulations: int = 800,
) -> str:
    """
    Helper to archive a list of GameRecord objects.

    Args:
        records: List of GameRecord from selfplay
        archive: GameArchive instance
        model_version: Model that generated these games
        simulations: MCTS simulations per move

    Returns:
        batch_id
    """
    all_examples = []
    game_results = []

    for record in records:
        examples = record.training_examples(use_ball_shaping=False)
        all_examples.extend(examples)
        game_results.append(record.result)

    if not all_examples:
        raise ValueError("No examples to archive")

    states = np.array([e[0] for e in all_examples])
    policies = np.array([e[1] for e in all_examples])
    values = np.array([e[2] for e in all_examples])
    legal_masks = np.array([e[3] for e in all_examples])

    return archive.archive_games(
        states=states,
        policies=policies,
        values=values,
        legal_masks=legal_masks,
        model_version=model_version,
        num_games=len(records),
        game_results=game_results,
        simulations=simulations,
    )

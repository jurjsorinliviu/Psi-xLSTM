"""Chronological trajectory generation for Ψ-xLSTM experiments.

Every tensor exposed by this module has shape
``[trajectory, sequence_length, features]``. A trajectory is never flattened
into a sample table, and train/validation/test splitting is performed before
any batching or truncated backpropagation through time (TBPTT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


TensorSplit = Dict[str, torch.Tensor | tuple[str, ...]]
SequenceDataset = Dict[str, TensorSplit]


def require_sequence_tensor(
    value: torch.Tensor,
    *,
    name: str,
    feature_size: int | None = None,
) -> torch.Tensor:
    """Validate the canonical ``[batch, sequence, features]`` contract."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 3:
        raise ValueError(
            f"{name} must have shape [batch, sequence_length, features], "
            f"got {tuple(value.shape)}"
        )
    if value.shape[0] < 1 or value.shape[1] <= 1 or value.shape[2] < 1:
        raise ValueError(
            f"{name} requires a nonempty batch, sequence_length > 1, "
            "and at least one feature"
        )
    if feature_size is not None and value.shape[2] != feature_size:
        raise ValueError(
            f"{name} feature dimension must be {feature_size}, "
            f"got {value.shape[2]}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def validate_split(split: TensorSplit, *, name: str = "split") -> TensorSplit:
    """Validate one complete-trajectory dataset split."""
    required = {"t", "V", "I", "w", "trajectory_ids", "source_ids"}
    missing = required - set(split)
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    t = require_sequence_tensor(split["t"], name=f"{name}.t", feature_size=1)
    voltage = require_sequence_tensor(split["V"], name=f"{name}.V", feature_size=1)
    current = require_sequence_tensor(split["I"], name=f"{name}.I")
    state = require_sequence_tensor(split["w"], name=f"{name}.w", feature_size=1)
    if not (t.shape[:2] == voltage.shape[:2] == current.shape[:2] == state.shape[:2]):
        raise ValueError(f"{name} tensors must share batch and sequence dimensions")
    if bool(torch.any(t[:, 1:] <= t[:, :-1])):
        raise ValueError(f"{name} timesteps must be strictly increasing")
    trajectory_ids = tuple(split["trajectory_ids"])
    source_ids = tuple(split["source_ids"])
    if len(trajectory_ids) != t.shape[0] or len(source_ids) != t.shape[0]:
        raise ValueError(f"{name} requires one trajectory/source ID per trajectory")
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError(f"{name} trajectory IDs must be unique")
    return split


def validate_dataset(dataset: SequenceDataset) -> SequenceDataset:
    """Validate all splits and prove that source identities are disjoint."""
    if set(dataset) != {"train", "val", "test"}:
        raise ValueError("dataset must contain exactly train, val, and test splits")
    for split_name, split in dataset.items():
        validate_split(split, name=split_name)
    identities = [set(dataset[name]["source_ids"]) for name in ("train", "val", "test")]
    for left in range(3):
        for right in range(left + 1, 3):
            overlap = identities[left] & identities[right]
            if overlap:
                raise ValueError(
                    "source identities must be disjoint across splits; "
                    f"overlap={sorted(overlap)}"
                )
    return dataset


def detach_state(state):
    """Detach nested recurrent state at a TBPTT boundary."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(detach_state(item) for item in state)
    if isinstance(state, list):
        return [detach_state(item) for item in state]
    raise TypeError(f"unsupported recurrent state type: {type(state)!r}")


def contiguous_chunks(
    V: torch.Tensor,
    t: torch.Tensor,
    I: torch.Tensor,
    chunk_length: int | None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield chronological TBPTT chunks without creating length-one chunks."""
    require_sequence_tensor(V, name="V")
    require_sequence_tensor(t, name="t", feature_size=1)
    require_sequence_tensor(I, name="I")
    if V.shape[:2] != t.shape[:2] or V.shape[:2] != I.shape[:2]:
        raise ValueError("V, t, and I must share batch and sequence dimensions")
    if chunk_length is None:
        yield V, t, I
        return
    if chunk_length <= 1:
        raise ValueError("chunk_length must be greater than one")
    start = 0
    length = V.shape[1]
    while start < length:
        stop = min(start + chunk_length, length)
        if length - stop == 1:
            stop = length
        yield V[:, start:stop], t[:, start:stop], I[:, start:stop]
        start = stop


@dataclass
class MemristorConfig:
    """Configuration for the VTEAM-inspired synthetic memristor model."""

    v_on: float = 0.5
    v_off: float = -0.5
    k_on: float = 8000.0
    k_off: float = -8000.0
    alpha_on: float = 3.0
    alpha_off: float = 3.0
    w_min: float = 1.0
    w_max: float = 10.0
    R_on: float = 100.0
    R_off: float = 10000.0
    dt: float = 1e-7
    t_max: float = 1e-3

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.t_max <= 0:
            raise ValueError("dt and t_max must be positive")
        self.num_steps = int(round(self.t_max / self.dt))
        if self.num_steps <= 1:
            raise ValueError("simulation must contain more than one timestep")


class MemristorDataGenerator:
    """Generate complete, independent chronological memristor trajectories."""

    def __init__(self, config: MemristorConfig | None = None, seed: int = 42):
        self.config = config or MemristorConfig()
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

    def generate_voltage_waveform(
        self,
        f_high: float = 50e3,
        f_low: float = 1e3,
        add_transients: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one ordered voltage trajectory."""
        t = np.arange(self.config.num_steps, dtype=np.float64) * self.config.dt
        voltage = 0.5 * np.sin(2 * np.pi * f_low * t)
        voltage += 0.3 * np.sin(2 * np.pi * f_high * t)
        if add_transients:
            for t_step in (0.2e-3, 0.5e-3, 0.8e-3):
                if t_step >= t[-1]:
                    continue
                mask = t > t_step
                voltage[mask] += 0.4 * np.exp(-1e4 * (t[mask] - t_step))
        return t, voltage

    def _vteam_switching_function(self, voltage: float, state: float) -> float:
        cfg = self.config
        if voltage > cfg.v_on:
            value = cfg.k_on * (cfg.w_max - state) * np.power(
                voltage / cfg.v_on - 1, cfg.alpha_on
            )
            return max(0.0, float(value))
        if voltage < cfg.v_off:
            value = cfg.k_off * state * np.power(
                voltage / cfg.v_off - 1, cfg.alpha_off
            )
            return min(0.0, float(value))
        return 0.0

    def _compute_current(self, voltage: float, state: float) -> float:
        cfg = self.config
        normalized = np.clip(
            (state - cfg.w_min) / (cfg.w_max - cfg.w_min), 0.0, 1.0
        )
        resistance = cfg.R_on * normalized + cfg.R_off * (1.0 - normalized)
        return float(voltage / resistance)

    def simulate_transient(
        self,
        V: np.ndarray,
        t: np.ndarray,
        w_init: float = 5.0,
        add_noise: bool = True,
        noise_level: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate one trajectory without resetting state inside it."""
        V = np.asarray(V, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if V.ndim != 1 or t.ndim != 1 or len(V) != len(t) or len(t) <= 1:
            raise ValueError("V and t must be equal-length one-dimensional sequences")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing")
        state = np.zeros(len(t), dtype=np.float64)
        current = np.zeros(len(t), dtype=np.float64)
        state[0] = np.clip(w_init, self.config.w_min, self.config.w_max)
        current[0] = self._compute_current(V[0], state[0])
        for index in range(1, len(t)):
            dt = t[index] - t[index - 1]
            derivative = self._vteam_switching_function(V[index - 1], state[index - 1])
            state[index] = np.clip(
                state[index - 1] + dt * derivative,
                self.config.w_min,
                self.config.w_max,
            )
            current[index] = self._compute_current(V[index], state[index])
        if add_noise:
            current = current + noise_level * np.std(current) * self.rng.randn(len(t))
        return current, state

    @staticmethod
    def _split_counts(
        num_sequences: int,
        split_ratio: Tuple[float, float, float],
    ) -> tuple[int, int, int]:
        if num_sequences < 3:
            raise ValueError("at least three complete trajectories are required")
        ratios = np.asarray(split_ratio, dtype=float)
        if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
            raise ValueError("split_ratio must contain three positive values summing to one")
        train = max(1, int(np.floor(num_sequences * ratios[0])))
        validation = max(1, int(np.floor(num_sequences * ratios[1])))
        while train + validation >= num_sequences:
            if train >= validation and train > 1:
                train -= 1
            elif validation > 1:
                validation -= 1
            else:
                raise ValueError("cannot create nonempty trajectory splits")
        return train, validation, num_sequences - train - validation

    def generate_dataset(
        self,
        num_sequences: int = 10,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        device: str | torch.device | None = None,
        *,
        f_high_start: float = 20e3,
        f_high_step: float = 10e3,
        f_low_start: float = 500.0,
        f_low_step: float = 500.0,
        noise_level: float = 0.01,
    ) -> SequenceDataset:
        """Generate and split complete trajectories.

        Sequence order is preserved. No timestep is shuffled and no source is
        allowed to cross a split boundary.
        """
        target_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        n_train, n_val, _ = self._split_counts(num_sequences, split_ratio)
        records: list[dict[str, np.ndarray | str]] = []
        for sequence_index in range(num_sequences):
            f_high = f_high_start + sequence_index * f_high_step
            f_low = f_low_start + sequence_index * f_low_step
            t, voltage = self.generate_voltage_waveform(f_high=f_high, f_low=f_low)
            initial_state = self.config.w_min + self.rng.rand() * (
                self.config.w_max - self.config.w_min
            )
            current, state = self.simulate_transient(
                voltage,
                t,
                w_init=initial_state,
                add_noise=True,
                noise_level=noise_level,
            )
            source_id = f"synthetic-memristor-{sequence_index:04d}"
            records.append(
                {
                    "t": t[:, None],
                    "V": voltage[:, None],
                    "I": current[:, None],
                    "w": state[:, None],
                    "trajectory_id": source_id,
                    "source_id": source_id,
                }
            )

        boundaries = {
            "train": records[:n_train],
            "val": records[n_train : n_train + n_val],
            "test": records[n_train + n_val :],
        }
        dataset: SequenceDataset = {}
        for split_name, items in boundaries.items():
            dataset[split_name] = {
                key: torch.tensor(
                    np.stack([item[key] for item in items]),
                    dtype=torch.float32,
                    device=target_device,
                )
                for key in ("t", "V", "I", "w")
            }
            dataset[split_name]["trajectory_ids"] = tuple(
                str(item["trajectory_id"]) for item in items
            )
            dataset[split_name]["source_ids"] = tuple(
                str(item["source_id"]) for item in items
            )
        validate_dataset(dataset)
        return dataset

    def get_physics_function(self):
        """Return the scalar simulator functions used to generate trajectories."""
        return self._vteam_switching_function, self._compute_current


if __name__ == "__main__":
    generated = MemristorDataGenerator().generate_dataset(num_sequences=5, device="cpu")
    for split_name, split in generated.items():
        print(split_name, tuple(split["V"].shape), split["trajectory_ids"])

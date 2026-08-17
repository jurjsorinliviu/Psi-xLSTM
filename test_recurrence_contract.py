"""Regression tests that prevent Psi-xLSTM from becoming pointwise again."""

from __future__ import annotations

import tempfile
import unittest

import torch

from data.memristor_generator import (
    MemristorConfig,
    MemristorDataGenerator,
    contiguous_chunks,
    detach_state,
    validate_dataset,
)
from models.clustering_student import ClusteringStudent
from models.lowrank_mlstm import LowRankMLSTM
from models.xlstm_teacher import xLSTMTeacher
from training.trainer import _loader, train_student, train_teacher


def sequence_pair(batch: int = 2, length: int = 8):
    time_values = (
        torch.arange(length, dtype=torch.float32)
        .view(1, length, 1)
        .repeat(batch, 1, 1)
    )
    voltage = torch.stack(
        [
            torch.linspace(-1.0 + index, 1.0 + index, length)
            for index in range(batch)
        ]
    ).unsqueeze(-1)
    return voltage, time_values


def state_tensors(state):
    if isinstance(state, torch.Tensor):
        return [state]
    values = []
    for item in state:
        values.extend(state_tensors(item))
    return values


class RecurrenceContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.models = (
            xLSTMTeacher(hidden_size=8, num_layers=2, num_heads=2),
            ClusteringStudent(hidden_size=8, num_layers=2, num_clusters=2),
            LowRankMLSTM(
                hidden_size=8, num_layers=2, rank=2, num_heads=2
            ),
        )

    def test_public_recurrent_models_reject_pointwise_and_length_one(self):
        point_voltage = torch.randn(4, 1)
        point_time = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
        short_voltage = point_voltage[:, None, :]
        short_time = point_time[:, None, :]
        for model in self.models:
            with self.subTest(model=model.__class__.__name__, shape="pointwise"):
                with self.assertRaises(ValueError):
                    model(point_voltage, point_time)
            with self.subTest(model=model.__class__.__name__, shape="length-one"):
                with self.assertRaises(ValueError):
                    model(short_voltage, short_time)

    def test_nonchronological_time_is_rejected(self):
        voltage, time_values = sequence_pair()
        time_values[:, [3, 4]] = time_values[:, [4, 3]]
        for model in self.models:
            with self.subTest(model=model.__class__.__name__):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    model(voltage, time_values)

    def test_batch_members_are_independent_and_default_state_resets(self):
        voltage, time_values = sequence_pair()
        for model in self.models:
            model.eval()
            with torch.no_grad():
                batched, _ = model(voltage, time_values)
                first, first_state = model(voltage[:1], time_values[:1])
                second, second_state = model(voltage[1:], time_values[1:])
                repeated_second, _ = model(voltage[1:], time_values[1:])
                _, leaked_state = model(
                    voltage[1:], time_values[1:], detach_state(first_state)
                )
            with self.subTest(model=model.__class__.__name__, property="batch"):
                self.assertTrue(torch.allclose(batched[:1], first, atol=1e-6))
                self.assertTrue(torch.allclose(batched[1:], second, atol=1e-6))
            with self.subTest(model=model.__class__.__name__, property="reset"):
                self.assertTrue(torch.equal(second, repeated_second))
                self.assertTrue(
                    any(
                        not torch.equal(reset_value, leaked_value)
                        for reset_value, leaked_value in zip(
                            state_tensors(second_state), state_tensors(leaked_state)
                        )
                    )
                )

    def test_contiguous_tbptt_matches_full_forward(self):
        voltage, time_values = sequence_pair(length=9)
        current = torch.zeros_like(voltage)
        for model in self.models:
            model.eval()
            with torch.no_grad():
                full, _ = model(voltage, time_values)
                state = None
                partial = []
                for voltage_chunk, time_chunk, _ in contiguous_chunks(
                    voltage, time_values, current, chunk_length=4
                ):
                    prediction, state = model(voltage_chunk, time_chunk, state)
                    partial.append(prediction)
                    state = detach_state(state)
                combined = torch.cat(partial, dim=1)
            with self.subTest(model=model.__class__.__name__):
                self.assertEqual(combined.shape, full.shape)
                self.assertTrue(torch.allclose(combined, full, atol=1e-6))

    def test_trajectory_splits_and_loader_preserve_internal_order(self):
        dataset = MemristorDataGenerator(
            MemristorConfig(dt=1e-4, t_max=8e-4), seed=9
        ).generate_dataset(num_sequences=6, device="cpu", noise_level=0.0)
        validate_dataset(dataset)
        source_sets = [
            set(dataset[name]["source_ids"]) for name in ("train", "val", "test")
        ]
        self.assertTrue(source_sets[0].isdisjoint(source_sets[1]))
        self.assertTrue(source_sets[0].isdisjoint(source_sets[2]))
        self.assertTrue(source_sets[1].isdisjoint(source_sets[2]))
        for voltage, time_values, current in _loader(
            dataset["train"],
            batch_size=2,
            shuffle_trajectories=True,
            seed=123,
        ):
            self.assertEqual(voltage.ndim, 3)
            self.assertEqual(current.ndim, 3)
            self.assertGreater(voltage.shape[1], 1)
            self.assertTrue(torch.all(time_values[:, 1:] > time_values[:, :-1]))

    def test_teacher_and_rrad_training_smoke(self):
        dataset = MemristorDataGenerator(
            MemristorConfig(dt=1e-4, t_max=8e-4), seed=11
        ).generate_dataset(num_sequences=6, device="cpu", noise_level=0.0)
        teacher = xLSTMTeacher(hidden_size=8, num_layers=2, num_heads=2)
        student = ClusteringStudent(
            hidden_size=4, num_layers=1, num_clusters=2
        )
        with tempfile.TemporaryDirectory() as directory:
            teacher, teacher_history = train_teacher(
                teacher,
                dataset,
                num_epochs=1,
                batch_size=2,
                device="cpu",
                save_dir=directory,
                chunk_length=4,
            )
            student, student_history = train_student(
                student,
                teacher,
                dataset,
                num_epochs=1,
                batch_size=2,
                device="cpu",
                save_dir=directory,
                chunk_length=4,
                update_clusters_interval=10,
            )
        self.assertEqual(len(teacher_history["val_loss"]), 1)
        self.assertEqual(len(student_history["val_loss"]), 1)
        self.assertTrue(torch.isfinite(torch.tensor(student_history["val_loss"])))


if __name__ == "__main__":
    unittest.main()

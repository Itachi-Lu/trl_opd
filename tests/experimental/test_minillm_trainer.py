# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer

from ..testing_utils import TrlTestCase


def make_support_loss_trainer(
    *,
    distill_mode: str = "student_topp_reverse_kl",
    support_top_p: float = 0.85,
    support_min_k: int = 2,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
):
    trainer = object.__new__(MiniLLMTrainer)
    trainer.distill_mode = distill_mode
    trainer.support_top_p = support_top_p
    trainer.support_min_k = support_min_k
    trainer.support_loss_use_teacher_mass_weighting = False
    trainer.support_loss_teacher_entropy_weighting = None
    trainer.pad_token_id = pad_token_id
    trainer.eos_token_id = eos_token_id
    return trainer


@pytest.mark.low_priority
class TestMiniLLMTrainer(TrlTestCase):
    def test_config_distill_mode_defaults_to_reverse_kl(self):
        config = MiniLLMConfig(output_dir=self.tmp_dir, report_to="none", num_generations=1, use_cpu=True)

        assert config.distill_mode == "reverse_kl"
        assert config.support_top_p == pytest.approx(0.95)
        assert config.support_min_k == 32
        assert config.support_loss_coef == pytest.approx(1.0)
        assert config.da_opd_weighting is False
        assert config.da_opd_tau == pytest.approx(2.0)
        assert config.opd_use_reward_advantage is False

    def test_train(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Initialize the trainer
        training_args = MiniLLMConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=32,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = MiniLLMTrainer(
            model="trl-internal-testing/small-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_loss_mask_uses_completion_and_tool_masks(self):
        trainer = object.__new__(MiniLLMTrainer)
        inputs = {
            "completion_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long),
            "tool_mask": torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.long),
        }

        mask = trainer._get_loss_mask(inputs)

        expected = torch.tensor([[True, False, False], [True, False, False]])
        assert torch.equal(mask, expected)

    def test_da_opd_weights_use_shifted_prefix_drift_and_mask(self):
        trainer = object.__new__(MiniLLMTrainer)
        trainer.da_opd_tau = 2.0
        teacher_logps = torch.tensor([[-1.0, -2.0, -3.0]], dtype=torch.float32)
        student_logps = torch.tensor([[-1.0, -4.0, -1.0]], dtype=torch.float32, requires_grad=True)
        mask = torch.tensor([[True, True, False]])

        weights, rho = trainer._compute_da_opd_token_weights(teacher_logps, student_logps, mask)

        expected_rho = torch.tensor([[0.0, 2.0, 2.0]], dtype=torch.float32)
        expected_weights = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
        assert torch.allclose(rho, expected_rho)
        assert torch.allclose(weights, expected_weights)
        assert weights.requires_grad is False

    def test_dual_gate_respects_padding_and_preserves_scale(self):
        trainer = object.__new__(MiniLLMTrainer)
        trainer.gate_teacher_entropy_lambda = 2.0
        trainer.gate_student_topk = 2
        trainer.gate_weight_min = 0.2
        trainer.gate_weight_max = 2.0

        student_logits = torch.tensor(
            [[[4.0, 3.0, 0.0, -1.0], [0.0, 3.0, 2.5, -1.0], [1.0, 0.0, -1.0, -2.0]]],
            dtype=torch.float32,
        )
        teacher_probs = torch.tensor(
            [[[0.82, 0.10, 0.05, 0.03], [0.50, 0.30, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25]]],
            dtype=torch.float32,
        )
        teacher_log_probs = teacher_probs.log()
        mask = torch.tensor([[True, True, False]])

        gate = trainer._compute_dual_gate(student_logits, teacher_log_probs, mask)

        assert gate.shape == mask.shape
        assert gate[0, 2].item() == 0.0
        assert gate[0, 0].item() > gate[0, 1].item()
        assert torch.isclose(gate[mask].mean(), torch.tensor(1.0), atol=1e-4)

    def test_student_topp_support_respects_top_p_and_min_k(self):
        trainer = object.__new__(MiniLLMTrainer)
        student_probs = torch.tensor(
            [[[0.70, 0.20, 0.07, 0.03], [0.50, 0.25, 0.15, 0.10]]],
            dtype=torch.float32,
        )
        student_logits = student_probs.log()

        topk_indices, support_sizes = trainer._build_student_topp_support(student_logits, top_p=0.85, min_k=3)

        expected_sizes = torch.tensor([[3, 3]], dtype=torch.long)
        assert torch.equal(support_sizes, expected_sizes)

        assert topk_indices.shape[-1] >= 3
        assert topk_indices[0, 0, 0].item() == 0
        assert topk_indices[0, 0, 1].item() == 1

    def test_support_reverse_kl_is_zero_when_teacher_matches_student(self):
        trainer = make_support_loss_trainer()

        student_logits = torch.log(
            torch.tensor(
                [[[0.55, 0.25, 0.15, 0.05], [0.45, 0.30, 0.20, 0.05]]],
                dtype=torch.float32,
            )
        )
        teacher_logits = student_logits.clone()
        mask = torch.tensor([[True, True]])

        loss = trainer._compute_support_distill_loss(student_logits, teacher_logits, mask)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_support_distill_loss_uses_da_opd_weighted_denominator(self):
        trainer = make_support_loss_trainer(support_top_p=1.0, support_min_k=2)
        student_logits = torch.log(
            torch.tensor(
                [[[0.80, 0.20], [0.60, 0.40], [0.50, 0.50]]],
                dtype=torch.float32,
            )
        )
        teacher_logits = torch.log(
            torch.tensor(
                [[[0.50, 0.50], [0.30, 0.70], [0.50, 0.50]]],
                dtype=torch.float32,
            )
        )
        mask = torch.tensor([[True, True, False]])
        token_weights = torch.tensor([[0.25, 0.75, 0.0]], dtype=torch.float32)

        loss = trainer._compute_support_distill_loss(student_logits, teacher_logits, mask, token_weights=token_weights)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        per_token_loss = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        expected = (per_token_loss * token_weights).sum() / token_weights.sum()
        assert torch.isclose(loss, expected, atol=1e-6)

    def test_support_filter_excludes_pad_and_eos_only_for_student_topp_reverse_kl(self):
        trainer = make_support_loss_trainer(pad_token_id=0, eos_token_id=1)
        support_indices = torch.tensor([[[0, 2, 1, 3]]], dtype=torch.long)
        local_support_mask = torch.tensor([[[True, True, True, False]]])

        filtered_indices, filtered_mask = trainer._exclude_stop_tokens_from_support(support_indices, local_support_mask)

        assert filtered_mask.sum().item() == 1
        assert filtered_indices[0, 0, 0].item() == 2
        assert 0 not in filtered_indices[filtered_mask].tolist()
        assert 1 not in filtered_indices[filtered_mask].tolist()

        reverse_kl_trainer = make_support_loss_trainer(
            distill_mode="reverse_kl",
            pad_token_id=0,
            eos_token_id=1,
        )
        unchanged_indices, unchanged_mask = reverse_kl_trainer._exclude_stop_tokens_from_support(
            support_indices, local_support_mask
        )
        assert torch.equal(unchanged_indices, support_indices)
        assert torch.equal(unchanged_mask, local_support_mask)

    def test_support_distill_loss_is_zero_when_filtered_support_is_empty(self):
        trainer = make_support_loss_trainer(
            support_top_p=0.95,
            support_min_k=2,
            pad_token_id=0,
            eos_token_id=1,
        )
        student_logits = torch.log(torch.tensor([[[0.60, 0.39, 0.01]]], dtype=torch.float32))
        teacher_logits = torch.log(torch.tensor([[[0.10, 0.80, 0.10]]], dtype=torch.float32))
        mask = torch.tensor([[True]])

        loss = trainer._compute_support_distill_loss(student_logits, teacher_logits, mask)

        assert torch.isfinite(loss)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_support_distill_reverse_kl_mode_does_not_require_old_topk_cache(self):
        class DummyCausalLM(nn.Module):
            def __init__(self, logits: torch.Tensor):
                super().__init__()
                self.register_buffer("fixed_logits", logits)

            def forward(self, input_ids=None, attention_mask=None, use_cache=False):
                batch_size = input_ids.size(0)
                logits = self.fixed_logits.expand(batch_size, -1, -1).clone()
                return type("DummyOutput", (), {"logits": logits})()

        class DummyAccelerator:
            device = torch.device("cpu")

            @staticmethod
            def gather(value):
                return value

            @staticmethod
            def pad_across_processes(value, dim=0, pad_index=0):
                return value

        trainer = object.__new__(MiniLLMTrainer)
        trainer.teacher_model = DummyCausalLM(
            torch.log(
                torch.tensor(
                    [[[0.40, 0.30, 0.20, 0.10], [0.45, 0.25, 0.20, 0.10], [0.50, 0.20, 0.20, 0.10]]],
                    dtype=torch.float32,
                )
            )
        )
        trainer.kd_temperature = 1.0
        trainer.distill_mode = "student_topp_reverse_kl"
        trainer.rkl_advantage = True
        trainer.single_step_decomposition = False
        trainer.support_top_p = 0.85
        trainer.support_min_k = 2
        trainer.support_loss_coef = 1.0
        trainer.support_loss_use_teacher_mass_weighting = False
        trainer.support_loss_teacher_entropy_weighting = None
        trainer.use_dual_gate = False
        trainer.da_opd_weighting = False
        trainer.opd_use_reward_advantage = False
        trainer.gamma = 0.0
        trainer.length_normalization = True
        trainer.accelerator = DummyAccelerator()
        trainer._metrics = defaultdict(lambda: defaultdict(list))
        trainer._log_teacher_mass_metrics = lambda *args, **kwargs: None
        trainer._log_teacher_entropy_metrics = lambda *args, **kwargs: None
        trainer._compute_loss = lambda model, inputs: torch.zeros((), dtype=torch.float32)

        model = DummyCausalLM(
            torch.log(
                torch.tensor(
                    [[[0.45, 0.30, 0.15, 0.10], [0.50, 0.20, 0.20, 0.10], [0.35, 0.30, 0.20, 0.15]]],
                    dtype=torch.float32,
                )
            )
        )
        inputs = {
            "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "completion_ids": torch.tensor([[3]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "completion_mask": torch.tensor([[1]], dtype=torch.long),
            "advantages": torch.tensor([0.0], dtype=torch.float32),
        }
        trainer.model = model

        loss = trainer.compute_loss(model, inputs)

        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_reverse_kl_da_opd_zeroes_reward_advantage_and_passes_token_weights(self):
        class DummyCausalLM(nn.Module):
            def __init__(self, logits: torch.Tensor):
                super().__init__()
                self.register_buffer("fixed_logits", logits)

            def forward(self, input_ids=None, attention_mask=None, use_cache=False):
                batch_size = input_ids.size(0)
                logits = self.fixed_logits.expand(batch_size, -1, -1).clone()
                return type("DummyOutput", (), {"logits": logits})()

        class DummyAccelerator:
            device = torch.device("cpu")

            @staticmethod
            def gather(value):
                return value

            @staticmethod
            def pad_across_processes(value, dim=0, pad_index=0):
                return value

        teacher_logits = torch.log(
            torch.tensor(
                [[[0.70, 0.20, 0.10], [0.60, 0.30, 0.10], [0.20, 0.70, 0.10], [0.20, 0.20, 0.60]]],
                dtype=torch.float32,
            )
        )
        student_logits = torch.log(
            torch.tensor(
                [[[0.60, 0.30, 0.10], [0.50, 0.40, 0.10], [0.30, 0.60, 0.10], [0.30, 0.20, 0.50]]],
                dtype=torch.float32,
            )
        )

        trainer = object.__new__(MiniLLMTrainer)
        trainer.teacher_model = DummyCausalLM(teacher_logits)
        trainer.kd_temperature = 1.0
        trainer.distill_mode = "reverse_kl"
        trainer.rkl_advantage = True
        trainer.single_step_decomposition = False
        trainer.support_loss_coef = 1.0
        trainer.use_dual_gate = False
        trainer.da_opd_weighting = True
        trainer.da_opd_tau = 2.0
        trainer.opd_use_reward_advantage = False
        trainer.gamma = 0.0
        trainer.length_normalization = True
        trainer.accelerator = DummyAccelerator()
        trainer._metrics = defaultdict(lambda: defaultdict(list))
        trainer._log_teacher_mass_metrics = lambda *args, **kwargs: None
        trainer._log_teacher_entropy_metrics = lambda *args, **kwargs: None
        trainer._compute_support_distill_loss = lambda *args, **kwargs: torch.zeros((), dtype=torch.float32)
        captured_inputs = {}

        def capture_compute_loss(model, inputs):
            captured_inputs.update(inputs)
            return torch.zeros((), dtype=torch.float32)

        trainer._compute_loss = capture_compute_loss
        model = DummyCausalLM(student_logits)
        inputs = {
            "prompt_ids": torch.tensor([[0, 1]], dtype=torch.long),
            "completion_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "completion_mask": torch.tensor([[1, 0]], dtype=torch.long),
            "advantages": torch.tensor([5.0], dtype=torch.float32),
            "old_per_token_logps": torch.tensor([[-0.8, -0.4]], dtype=torch.float32),
        }
        trainer.model = model

        loss = trainer.compute_loss(model, inputs)

        assert torch.isfinite(loss)
        assert "token_loss_weights" in captured_inputs
        assert captured_inputs["advantages"].shape == torch.Size([1, 2])
        expected_teacher_on_labels = torch.tensor([[torch.log(torch.tensor(0.30)), torch.log(torch.tensor(0.10))]])
        expected_reverse_kl_advantage = expected_teacher_on_labels - inputs["old_per_token_logps"]
        expected_reverse_kl_advantage = expected_reverse_kl_advantage * inputs["completion_mask"].float()
        assert torch.allclose(captured_inputs["advantages"], expected_reverse_kl_advantage)
        assert torch.allclose(captured_inputs["token_loss_weights"], torch.tensor([[0.5, 0.0]]))

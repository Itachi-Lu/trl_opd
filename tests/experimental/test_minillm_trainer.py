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

import pytest
import torch
from datasets import load_dataset

from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer

from ..testing_utils import TrlTestCase


@pytest.mark.low_priority
class TestMiniLLMTrainer(TrlTestCase):
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


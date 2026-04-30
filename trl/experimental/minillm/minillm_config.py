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

from dataclasses import dataclass, field
from typing import Any

from transformers import TrainingArguments

from ...trainer.grpo_config import GRPOConfig


@dataclass
class MiniLLMConfig(GRPOConfig):
    """
    Configuration class for [`MiniLLMTrainer`].

    This class includes only the parameters that are specific to MiniLLM training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] and [`GRPOConfig`] documentation.

    Args:
        teacher_model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        rkl_advantage (`bool`, *optional*, defaults to `True`):
            Whether to add the reverse KL advantage to the reward advantage.
        single_step_decomposition (`bool`, *optional*, defaults to `True`):
            Whether to use single-step decomposition for the KL divergence computation.
        kd_temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for knowledge distillation. Higher temperatures produce softer probability distributions over
            classes.
        gamma (`float`, *optional*, defaults to `0.0`):
            Discount factor for future rewards in reinforcement learning.
        length_normalization (`bool`, *optional*, defaults to `True`):
            Whether to apply length normalization to the rewards.
        distill_mode (`str`, *optional*, defaults to `"reverse_kl"`):
            Distillation objective to use. `"reverse_kl"` keeps the existing reverse-KL advantage path.
            `"student_topp_reverse_kl"` optimizes reverse KL on the student's top-p/min-k support.
        support_top_p (`float`, *optional*, defaults to `0.95`):
            Cumulative probability threshold for the student's support-distillation set.
        support_min_k (`int`, *optional*, defaults to `32`):
            Minimum number of tokens to keep in the student's support-distillation set.
        support_loss_coef (`float`, *optional*, defaults to `1.0`):
            Coefficient applied to the direct support-distillation loss.
        support_loss_use_teacher_mass_weighting (`bool`, *optional*, defaults to `False`):
            Whether to weight each token's support distillation KL by the teacher probability mass assigned to the
            student's support tokens at that position.
        support_loss_teacher_entropy_weighting (`str | None`, *optional*, defaults to `None`):
            Optional teacher-entropy weighting applied after teacher-mass weighting. Supported values are `None`,
            `"teacher_entropy_full_vocab"`, and `"teacher_entropy_support"`.
        use_dual_gate (`bool`, *optional*, defaults to `False`):
            Whether to apply dual teacher/student token gating to the distillation advantage.
        use_gate_bonus (`bool`, *optional*, defaults to `True`):
            Whether to apply the disagreement-based bonus on top of the base gate.
        gate_teacher_entropy_lambda (`float`, *optional*, defaults to `2.0`):
            Entropy penalty strength for the teacher reliability gate.
        gate_student_topk (`int`, *optional*, defaults to `32`):
            Top-k size used to estimate whether the teacher's probability mass lies in the student's learnable region.
        gate_rank_tau (`float`, *optional*, defaults to `2.0`):
            Exponential decay temperature for rank-weighted coverage inside the student's top-k support.
        gate_weight_min (`float`, *optional*, defaults to `0.2`):
            Lower clip bound applied to the normalized base gate.
        gate_weight_max (`float`, *optional*, defaults to `2.0`):
            Upper clip bound applied to the normalized base gate.
        gate_bonus_min (`float`, *optional*, defaults to `1.0`):
            Lower clip bound applied to the normalized disagreement bonus.
        gate_bonus_max (`float`, *optional*, defaults to `2.0`):
            Upper clip bound applied to the normalized disagreement bonus.
    """

    _VALID_DICT_FIELDS = GRPOConfig._VALID_DICT_FIELDS + ["teacher_model_init_kwargs"]

    teacher_model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "teacher model from a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )
    rkl_advantage: bool = field(
        default=True,
        metadata={"help": "Whether to add the reverse KL advantage to the reward advantage."},
    )
    single_step_decomposition: bool = field(
        default=True,
        metadata={"help": "Whether to use single-step decomposition for the KL divergence computation."},
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for knowledge distillation. Higher temperatures produce softer probability "
            "distributions over classes."
        },
    )
    gamma: float = field(
        default=0.0,
        metadata={"help": "Discount factor for future rewards in reinforcement learning."},
    )
    length_normalization: bool = field(
        default=True,
        metadata={"help": "Whether to apply length normalization to the rewards."},
    )
    distill_mode: str = field(
        default="reverse_kl",
        metadata={
            "help": "Distillation objective to use. Supported values are 'reverse_kl' and "
            "'student_topp_reverse_kl'."
        },
    )
    support_top_p: float = field(
        default=0.95,
        metadata={"help": "Cumulative probability threshold for the student's support-distillation set."},
    )
    support_min_k: int = field(
        default=32,
        metadata={"help": "Minimum number of tokens to keep in the student's support-distillation set."},
    )
    support_loss_coef: float = field(
        default=1.0,
        metadata={"help": "Coefficient applied to the direct support-distillation loss."},
    )
    support_loss_use_teacher_mass_weighting: bool = field(
        default=False,
        metadata={
            "help": "Whether to weight each token's support distillation KL by the teacher mass on the student's "
            "support."
        },
    )
    support_loss_teacher_entropy_weighting: str | None = field(
        default=None,
        metadata={
            "help": "Optional entropy-based support-loss weighting. Supported values are None, "
            "'teacher_entropy_full_vocab', and 'teacher_entropy_support'."
        },
    )

    use_dual_gate: bool = field(
        default=False,
        metadata={"help": "Whether to apply dual teacher/student token gating to the distillation advantage."},
    )
    use_gate_bonus: bool = field(
        default=True,
        metadata={"help": "Whether to apply the disagreement-based bonus on top of the base gate."},
    )
    gate_teacher_entropy_lambda: float = field(
        default=2.0,
        metadata={"help": "Entropy penalty strength for the teacher reliability gate."},
    )
    gate_student_topk: int = field(
        default=32,
        metadata={"help": "Top-k size used to estimate whether the teacher's mass lies in the student's learnable region."},
    )
    gate_rank_tau: float = field(
        default=2.0,
        metadata={"help": "Exponential decay temperature for rank-weighted coverage inside the student's top-k support."},
    )
    gate_weight_min: float = field(
        default=0.2,
        metadata={"help": "Lower clip bound applied to the normalized base gate."},
    )
    gate_weight_max: float = field(
        default=2.0,
        metadata={"help": "Upper clip bound applied to the normalized base gate."},
    )
    gate_bonus_min: float = field(
        default=1.0,
        metadata={"help": "Lower clip bound applied to the normalized disagreement bonus."},
    )
    gate_bonus_max: float = field(
        default=2.0,
        metadata={"help": "Upper clip bound applied to the normalized disagreement bonus."},
    )

    def __post_init__(self):
        # We do not use the post_init of GRPOConfig because:
        # 1. num_generations can be < 2 in MiniLLMConfig. Scale_rewards must be set to "none" to avoid nan.
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        TrainingArguments.__post_init__(self)

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)
        if self.num_generations == 1:
            self.scale_rewards = "none"

        valid_distill_modes = {"reverse_kl", "student_topp_reverse_kl"}
        if self.distill_mode not in valid_distill_modes:
            raise ValueError(
                "distill_mode must be one of {'reverse_kl', 'student_topp_reverse_kl'}, but got "
                f"{self.distill_mode!r}."
            )
        if not (0.0 < self.support_top_p <= 1.0):
            raise ValueError(f"support_top_p must be in (0, 1], but got {self.support_top_p}.")
        if self.support_min_k < 1:
            raise ValueError(f"support_min_k must be >= 1, but got {self.support_min_k}.")
        if self.support_loss_coef < 0.0:
            raise ValueError(f"support_loss_coef must be >= 0, but got {self.support_loss_coef}.")
        if isinstance(self.support_loss_teacher_entropy_weighting, str):
            if self.support_loss_teacher_entropy_weighting.lower() == "none":
                self.support_loss_teacher_entropy_weighting = None
        valid_support_loss_teacher_entropy_weightings = {
            None,
            "teacher_entropy_full_vocab",
            "teacher_entropy_support",
        }
        if self.support_loss_teacher_entropy_weighting not in valid_support_loss_teacher_entropy_weightings:
            raise ValueError(
                "support_loss_teacher_entropy_weighting must be one of "
                "{None, 'teacher_entropy_full_vocab', 'teacher_entropy_support'}, but got "
                f"{self.support_loss_teacher_entropy_weighting!r}."
            )

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Determine the number of generations to use for evaluation
            num_generations = self.num_generations_eval or self.num_generations

            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.delta is not None and self.use_liger_kernel:
            raise ValueError("Liger kernel does not support two-sided GRPO loss yet.")

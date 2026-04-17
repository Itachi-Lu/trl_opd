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

import math
import textwrap
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import Dataset, IterableDataset
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from ...models import prepare_deepspeed
from ...trainer.grpo_trainer import GRPOTrainer, RewardFunc, RolloutFunc
from ...trainer.utils import disable_dropout_in_model, get_config_model_id
from ..utils import empty_cache
from .minillm_config import MiniLLMConfig


if is_peft_available():
    from peft import PeftConfig


def dummy_reward_func(completions: list, **kwargs):
    # placeholder reward function when no reward function is provided
    return [1.0 for _ in completions]


class MiniLLMTrainer(GRPOTrainer):
    """
    Trainer for the Knowledge Distillation of Language Models (MiniLLM) method. This algorithm was initially proposed
    in the paper [Knowledge Distillation of Large Language Models](https://huggingface.co/papers/2306.08543).

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental.minillm import MiniLLMTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = MiniLLMTrainer(
        model="Qwen/Qwen3-0.6B",
        teacher_model="Qwen/Qwen3-1.7B",
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        teacher_model (`PreTrainedModel | nn.Module | str`):
            Teacher model used for knowledge distillation. Instantiated similarly to `model`.
        reward_funcs (`RewardFunc | list[RewardFunc]`, *optional*):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`experimental.minillm.MiniLLMConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It must take prompts, args, and processing_class as parameters
            and return a dict with `"prompt_ids"`, `"completion_ids"`, and `"logprobs"` fields. Any other fields that
            are forwarded to the reward functions. This feature is experimental and may change or be removed at any
            time without prior notice.
    """

    _tag_names = ["trl", "minillm"]
    _name = "MiniLLM"
    _paper = {
        "title": "MiniLLM: Knowledge Distillation of Large Language Models",
        "id": "2306.08543",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{
                gu2024minillm,
                title={{MiniLLM: Knowledge Distillation of Large Language Models}},
                author={Yuxian Gu and Li Dong and Furu Wei and Minlie Huang},
                booktitle={The Twelfth International Conference on Learning Representations},
                year={2024},
                url={https://openreview.net/forum?id=5h0qf7IBZZ}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        teacher_model: PreTrainedModel | nn.Module | str,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: MiniLLMConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
    ):
        if reward_funcs is None:
            reward_funcs = [dummy_reward_func]

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = MiniLLMConfig(f"{model_name}-MiniLLM")

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model,
            reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            rollout_func=rollout_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the MiniLLMConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.temperature = args.temperature
        self.kd_temperature = args.kd_temperature
        self.single_step_decomposition = args.single_step_decomposition
        self.rkl_advantage = args.rkl_advantage
        self.gamma = args.gamma
        self.length_normalization = args.length_normalization
        self.distill_mode = args.distill_mode
        self.support_top_p = args.support_top_p
        self.support_min_k = args.support_min_k
        self.support_loss_coef = args.support_loss_coef
        self.support_loss_use_teacher_mass_weighting = args.support_loss_use_teacher_mass_weighting
        self.use_dual_gate = args.use_dual_gate
        self.use_gate_bonus = args.use_gate_bonus
        self.gate_teacher_entropy_lambda = args.gate_teacher_entropy_lambda
        self.gate_student_topk = args.gate_student_topk
        self.gate_rank_tau = args.gate_rank_tau
        self.gate_weight_min = args.gate_weight_min
        self.gate_weight_max = args.gate_weight_max
        self.gate_bonus_min = args.gate_bonus_min
        self.gate_bonus_max = args.gate_bonus_max

    def _single_step_decomposition_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "batchmean",
    ):
        """
        Compute the MiniLLM loss for knowledge distillation using F.kl_div. See Eq. (1) of
        https://huggingface.co/papers/2306.08543 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """
        reg_loss = F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        )  # (batch_size, sequence_length)

        # Masking
        if mask is not None:
            reg_loss = reg_loss[mask]

        # Apply reduction
        if reduction == "batchmean":
            return reg_loss.sum() / mask.sum() if mask is not None else reg_loss.sum() / reg_loss.size(0)
        elif reduction == "sum":
            return reg_loss.sum()
        elif reduction == "mean":
            return reg_loss.mean()
        else:
            return reg_loss

    def _compute_advantage(
        self,
        student_log_probs_on_labels: torch.Tensor,
        teacher_log_probs_on_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Compute the advantage for Reverse KL Divergence.

        Mostly following [this
        implementation](https://github.com/microsoft/LMOps/blob/e210d2c026b9958617887762400778ace81172e6/minillm/minillm/losses.py#L37-L49).

        $$ \text{rewards}_t = \text{teacher\_log\_probs\_on\_labels}_t - \text{student\_log\_probs\_on\_labels}_t $$

        If length normalization is enabled:

        $$ \text{lengths}_t = \sum_{i=t}^{T} \gamma^{i-t} $$

        $$ \text{advantages}_t = \frac{\sum_{i=t}^{T} \gamma^{i-t} R_i}{\text{lengths}_t} $$

        Otherwise:

        $$ \text{advantages}_t = \sum_{i=t}^{T} \gamma^{i-t} R_i $$

        Args:
            student_log_probs_on_labels: Log probabilities of the student model on the labels.
                Shape: (batch_size, sequence_length)
            teacher_log_probs_on_labels: Log probabilities of the teacher model on the labels.
                Shape: (batch_size, sequence_length)
            mask: Optional mask to apply to the log probabilities. Shape: (batch_size, sequence_length)
        Returns:
            advantage: Computed advantage. Shape: (batch_size, sequence_length)
        """
        response_length = student_log_probs_on_labels.size(1)
        if mask is None:
            mask = torch.ones_like(student_log_probs_on_labels)
        mask = mask.float()
        student_log_probs_on_labels = student_log_probs_on_labels * mask
        teacher_log_probs_on_labels = teacher_log_probs_on_labels * mask

        rewards = teacher_log_probs_on_labels - student_log_probs_on_labels  # (batch_size, sequence_length)

        if self.gamma > 0.0:
            gamma_pow = torch.pow(self.gamma, torch.arange(response_length, device=rewards.device))

            advantages = rewards * gamma_pow
            advantages = advantages.flip(1).cumsum(dim=1).flip(1)

            if self.length_normalization:
                mask = torch.where(mask < 0.5, 1e-4, mask)
                lengths = mask * gamma_pow
                lengths = lengths.flip(1).cumsum(dim=1).flip(1)
                advantages = advantages / lengths
        else:
            advantages = rewards

        return advantages

    @staticmethod
    def _get_loss_mask(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = inputs["completion_mask"].bool()
        if "tool_mask" in inputs:
            mask = mask & inputs["tool_mask"].bool()
        return mask

    def _build_student_topp_support(
        self, student_logits: torch.Tensor, top_p: float, min_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_size = student_logits.size(-1)
        min_k = min(max(1, min_k), vocab_size)
        candidate_k = min(max(min_k, 512), vocab_size)

        student_probs = F.softmax(student_logits.detach(), dim=-1)
        topk_probs, topk_indices = student_probs.topk(candidate_k, dim=-1, sorted=True)
        cumulative_probs = topk_probs.cumsum(dim=-1)

        support_sizes = (cumulative_probs < top_p).sum(dim=-1) + 1
        support_sizes = support_sizes.clamp(min=min_k, max=candidate_k)

        return topk_indices, support_sizes

    @staticmethod
    def _build_local_support_mask(
        support_sizes: torch.Tensor, support_width: int, device: torch.device
    ) -> torch.Tensor:
        rank_positions = torch.arange(support_width, device=device).view(1, 1, support_width)
        return rank_positions < support_sizes.unsqueeze(-1)

    @staticmethod
    def _compute_teacher_mass(
        teacher_log_probs: torch.Tensor, support_indices: torch.Tensor, local_support_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        teacher_support_probs = teacher_log_probs.gather(dim=-1, index=support_indices).exp()
        if local_support_mask is not None:
            teacher_support_probs = teacher_support_probs * local_support_mask.to(teacher_support_probs.dtype)
        return teacher_support_probs.sum(dim=-1)

    def _log_masked_distribution_metrics(
        self, prefix: str, values: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> None:
        values = values.detach().float()
        gathered_values = self.accelerator.gather(values)
        gathered_mask = self.accelerator.gather(mask.detach().to(dtype=torch.int32)).bool()
        valid_values = gathered_values[gathered_mask]

        if valid_values.numel() == 0:
            zero = torch.zeros((), device=values.device)
            metric_tensors = {"mean": zero, "p10": zero, "p90": zero}
        else:
            metric_tensors = {
                "mean": valid_values.mean(),
                "p10": torch.quantile(valid_values, 0.1),
                "p90": torch.quantile(valid_values, 0.9),
            }

        for suffix, value in metric_tensors.items():
            self._metrics[mode][f"{prefix}/{suffix}"].append(value.item())

    def _log_teacher_mass_metrics(
        self, student_logits: torch.Tensor, teacher_log_probs: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> None:
        with torch.no_grad():
            max_topk = min(32, student_logits.size(-1))
            topk_indices = student_logits.detach().topk(max_topk, dim=-1).indices
            for k in (8, 16, 32):
                teacher_mass = self._compute_teacher_mass(teacher_log_probs, topk_indices[..., : min(k, max_topk)])
                self._log_masked_distribution_metrics(f"teacher_mass/topk_{k}", teacher_mass, mask, mode)

            for top_p, suffix in ((0.8, "80"), (0.9, "90")):
                support_indices, support_sizes = self._build_student_topp_support(
                    student_logits, top_p=top_p, min_k=1
                )
                local_support_mask = self._build_local_support_mask(
                    support_sizes, support_indices.size(-1), student_logits.device
                )
                teacher_mass = self._compute_teacher_mass(teacher_log_probs, support_indices, local_support_mask)
                self._log_masked_distribution_metrics(f"teacher_mass/topp_{suffix}", teacher_mass, mask, mode)

            support_indices, support_sizes = self._build_student_topp_support(
                student_logits, top_p=self.support_top_p, min_k=self.support_min_k
            )
            local_support_mask = self._build_local_support_mask(
                support_sizes, support_indices.size(-1), student_logits.device
            )
            teacher_mass = self._compute_teacher_mass(teacher_log_probs, support_indices, local_support_mask)
            self._log_masked_distribution_metrics("teacher_mass/support", teacher_mass, mask, mode)

    def _compute_support_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        topk_indices, support_sizes = self._build_student_topp_support(
            student_logits, top_p=self.support_top_p, min_k=self.support_min_k
        )
        max_support_size = int(support_sizes.max().item())
        support_indices = topk_indices[..., :max_support_size]
        local_support_mask = self._build_local_support_mask(support_sizes, max_support_size, student_logits.device)
        neg_large = torch.finfo(student_logits.dtype).min

        student_support_logits = student_logits.gather(dim=-1, index=support_indices)
        teacher_support_logits = teacher_logits.gather(dim=-1, index=support_indices)
        masked_student_support_logits = student_support_logits.masked_fill(~local_support_mask, neg_large)
        masked_teacher_support_logits = teacher_support_logits.masked_fill(~local_support_mask, neg_large)

        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_mass = self._compute_teacher_mass(teacher_log_probs, support_indices, local_support_mask)

        local_student_log_probs = F.log_softmax(masked_student_support_logits, dim=-1)
        local_teacher_log_probs = F.log_softmax(masked_teacher_support_logits, dim=-1)
        local_student_probs = local_student_log_probs.exp()

        per_token_loss = (local_student_probs * (local_student_log_probs - local_teacher_log_probs)).sum(dim=-1)
        if self.support_loss_use_teacher_mass_weighting:
            per_token_loss = per_token_loss * teacher_mass

        mask_float = mask.float()
        return (per_token_loss * mask_float).sum() / mask_float.sum().clamp(min=1.0)

    def _compute_dual_gate(
        self,
        student_logits: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        mask: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        teacher_probs = teacher_log_probs.exp()
        student_probs = F.softmax(student_logits, dim=-1)
        vocab_size = teacher_probs.size(-1)
        entropy_scale = math.log(vocab_size) if vocab_size > 1 else 1.0
        teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1) / entropy_scale
        teacher_gate = torch.exp(-self.gate_teacher_entropy_lambda * teacher_entropy)

        topk = max(1, min(self.gate_student_topk, student_logits.size(-1)))
        topk_indices = student_logits.topk(topk, dim=-1).indices
        teacher_slice = teacher_probs.gather(dim=-1, index=topk_indices)
        student_slice = student_probs.gather(dim=-1, index=topk_indices)

        rank_positions = torch.arange(topk, device=student_logits.device, dtype=student_logits.dtype)
        rank_tau = max(self.gate_rank_tau, 1e-6)
        rank_weights = torch.exp(-rank_positions / rank_tau)
        rank_weights = rank_weights / rank_weights.sum().clamp(min=1e-6)
        student_gate = (teacher_slice * rank_weights.view(1, 1, -1)).sum(dim=-1)

        teacher_local = teacher_slice / teacher_slice.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        student_local = student_slice / student_slice.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        disagreement = 0.5 * torch.abs(teacher_local - student_local).sum(dim=-1)

        mask_float = mask.float()
        valid_count = mask_float.sum().clamp(min=1.0)

        base_gate_raw = teacher_gate * student_gate
        mean_base_gate = (base_gate_raw * mask_float).sum() / valid_count
        base_gate_norm = base_gate_raw / (mean_base_gate + 1e-6)
        base_gate_low_clipped = base_gate_norm < self.gate_weight_min
        base_gate_high_clipped = base_gate_norm > self.gate_weight_max
        base_gate = base_gate_norm.clamp(min=self.gate_weight_min, max=self.gate_weight_max)

        if self.use_gate_bonus:
            mean_disagreement = (disagreement * mask_float).sum() / valid_count
            bonus_raw = disagreement / (mean_disagreement + 1e-6)
            bonus_low_clipped = bonus_raw < self.gate_bonus_min
            bonus_high_clipped = bonus_raw > self.gate_bonus_max
            bonus = bonus_raw.clamp(min=self.gate_bonus_min, max=self.gate_bonus_max)
        else:
            bonus_raw = torch.ones_like(disagreement)
            bonus_low_clipped = torch.zeros_like(disagreement, dtype=torch.bool)
            bonus_high_clipped = torch.zeros_like(disagreement, dtype=torch.bool)
            bonus = torch.ones_like(disagreement)

        final_gate = base_gate * bonus * mask_float

        if not return_stats:
            return final_gate

        stats = {
            "teacher_entropy": teacher_entropy,
            "teacher_gate": teacher_gate,
            "student_gate": student_gate,
            "disagreement": disagreement,
            "bonus": bonus,
            "bonus_low_clipped": bonus_low_clipped.float(),
            "bonus_high_clipped": bonus_high_clipped.float(),
            "base_gate": base_gate,
            "base_gate_low_clipped": base_gate_low_clipped.float(),
            "base_gate_high_clipped": base_gate_high_clipped.float(),
            "dual_gate_raw": base_gate_raw,
            "dual_gate_final": final_gate,
        }
        return final_gate, stats

    def _write_sample_rollout_file(self, checkpoint_dir: Path) -> None:
        sample_path = checkpoint_dir / "sample_rollout.txt"
        logs = getattr(self, "_logs", {})
        prompts = logs.get("prompt", [])
        prompts_with_special_tokens = logs.get("prompt_with_special_tokens", [])
        completions = logs.get("completion", [])
        completions_with_special_tokens = logs.get("completion_with_special_tokens", [])

        if len(prompts) == 0 or len(completions) == 0:
            sample_path.write_text("No rollout sample available at checkpoint time.\n", encoding="utf-8")
            return

        sample_idx = len(prompts) - 1

        def format_value(value):
            if isinstance(value, float):
                return f"{value:.6f}"
            if isinstance(value, list):
                if value and isinstance(value[0], list):
                    return "\n".join(str(item) for item in value)
                return ", ".join(str(item) for item in value)
            return str(value)

        prompt = prompts[sample_idx]
        prompt_with_special_tokens = (
            prompts_with_special_tokens[sample_idx] if len(prompts_with_special_tokens) > sample_idx else None
        )
        completion = completions[sample_idx]
        completion_with_special_tokens = (
            completions_with_special_tokens[sample_idx]
            if len(completions_with_special_tokens) > sample_idx
            else None
        )
        advantages = logs.get("advantages", [])
        advantage = advantages[sample_idx] if len(advantages) > sample_idx else "N/A"

        lines = [
            f"step: {self.state.global_step}",
            f"checkpoint: {checkpoint_dir.name}",
            "",
            "prompt",
            "------",
            format_value(prompt),
            "",
            "prompt_with_special_tokens",
            "--------------------------",
            format_value(prompt_with_special_tokens if prompt_with_special_tokens is not None else "N/A"),
            "",
            "completion",
            "----------",
            format_value(completion),
            "",
            "completion_with_special_tokens",
            "------------------------------",
            format_value(completion_with_special_tokens if completion_with_special_tokens is not None else "N/A"),
            "",
            "advantage",
            "---------",
            format_value(advantage),
        ]

        rewards = logs.get("rewards", {})
        if rewards:
            lines.extend(["", "rewards", "-------"])
            for name, values in rewards.items():
                if len(values) > sample_idx:
                    lines.append(f"{name}: {format_value(values[sample_idx])}")

        extra = logs.get("extra", {})
        if extra:
            extra_lines = []
            for name, values in extra.items():
                if len(values) > sample_idx:
                    extra_lines.append(f"{name}: {format_value(values[sample_idx])}")
            if extra_lines:
                lines.extend(["", "extra", "-----", *extra_lines])

        sample_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)

        # Compute student output
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompt_ids"].shape[1]
        student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = input_ids[:, prompt_lengths:]

        # Apply temperature scaling
        student_logits = student_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        student_log_probs_on_labels = torch.gather(
            student_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)
        teacher_log_probs_on_labels = torch.gather(
            teacher_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)

        mask = self._get_loss_mask(inputs)
        mode = "train" if self.model.training else "eval"
        self._log_teacher_mass_metrics(student_logits, teacher_log_probs, mask, mode)

        if self.distill_mode == "reverse_kl" and self.rkl_advantage:
            rollout_log_probs_on_labels = inputs.get("old_per_token_logps")
            if rollout_log_probs_on_labels is None:
                raise RuntimeError(
                    "MiniLLM reverse KL advantage requires `old_per_token_logps`, but it was not found in the inputs."
                )
            reverse_kl_teacher_log_probs = teacher_log_probs_on_labels

            reverse_kl_advantage = self._compute_advantage(
                student_log_probs_on_labels=rollout_log_probs_on_labels,
                teacher_log_probs_on_labels=reverse_kl_teacher_log_probs,
                mask=mask,
            )

            if self.use_dual_gate:
                dual_gate, gate_stats = self._compute_dual_gate(
                    student_logits, teacher_log_probs, mask, return_stats=True
                )
                reverse_kl_advantage = reverse_kl_advantage * dual_gate

                mask_float = mask.float()
                valid_count = mask_float.sum().clamp(min=1.0)
                valid_mask = mask.bool()
                raw_valid = gate_stats["dual_gate_raw"][valid_mask].float()
                base_valid = gate_stats["base_gate"][valid_mask].float()
                disagreement_valid = gate_stats["disagreement"][valid_mask].float()
                bonus_valid = gate_stats["bonus"][valid_mask].float()
                final_valid = gate_stats["dual_gate_final"][valid_mask].float()

                def masked_mean(x: torch.Tensor) -> torch.Tensor:
                    return (x * mask_float).sum() / valid_count

                def safe_std(x: torch.Tensor) -> torch.Tensor:
                    if x.numel() == 0:
                        return torch.zeros((), device=mask.device)
                    return x.std(unbiased=False)

                def safe_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
                    if x.numel() == 0:
                        return torch.zeros((), device=mask.device)
                    return torch.quantile(x, q)

                metric_tensors = {
                    "teacher_entropy/mean": masked_mean(gate_stats["teacher_entropy"]),
                    "teacher_gate/mean": masked_mean(gate_stats["teacher_gate"]),
                    "student_gate/mean": masked_mean(gate_stats["student_gate"]),
                    "student_gate/std": safe_std(gate_stats["student_gate"][valid_mask].float()),
                    "student_gate/p10": safe_quantile(gate_stats["student_gate"][valid_mask].float(), 0.1),
                    "student_gate/p90": safe_quantile(gate_stats["student_gate"][valid_mask].float(), 0.9),
                    "disagreement/mean": masked_mean(gate_stats["disagreement"]),
                    "disagreement/std": safe_std(disagreement_valid),
                    "disagreement/p10": safe_quantile(disagreement_valid, 0.1),
                    "disagreement/p90": safe_quantile(disagreement_valid, 0.9),
                    "bonus/mean": masked_mean(gate_stats["bonus"]),
                    "bonus/std": safe_std(bonus_valid),
                    "bonus/p10": safe_quantile(bonus_valid, 0.1),
                    "bonus/p90": safe_quantile(bonus_valid, 0.9),
                    "bonus/clipped_low_ratio": masked_mean(gate_stats["bonus_low_clipped"]),
                    "bonus/clipped_high_ratio": masked_mean(gate_stats["bonus_high_clipped"]),
                    "base_gate/mean": masked_mean(gate_stats["base_gate"]),
                    "base_gate/std": safe_std(base_valid),
                    "base_gate/p10": safe_quantile(base_valid, 0.1),
                    "base_gate/p90": safe_quantile(base_valid, 0.9),
                    "base_gate/clipped_low_ratio": masked_mean(gate_stats["base_gate_low_clipped"]),
                    "base_gate/clipped_high_ratio": masked_mean(gate_stats["base_gate_high_clipped"]),
                    "dual_gate/raw_mean": masked_mean(gate_stats["dual_gate_raw"]),
                    "dual_gate/raw_std": safe_std(raw_valid),
                    "dual_gate/raw_p10": safe_quantile(raw_valid, 0.1),
                    "dual_gate/raw_p90": safe_quantile(raw_valid, 0.9),
                    "dual_gate/mean": masked_mean(gate_stats["dual_gate_final"]),
                    "dual_gate/final_std": safe_std(final_valid),
                    "dual_gate/final_p10": safe_quantile(final_valid, 0.1),
                    "dual_gate/final_p90": safe_quantile(final_valid, 0.9),
                }
                for name, value in metric_tensors.items():
                    self._metrics[mode][name].append(self.accelerator.gather(value).nanmean().item())

            reverse_kl_advantage = reverse_kl_advantage.detach()
            inputs["advantages"] = inputs["advantages"].unsqueeze(1) + reverse_kl_advantage

        # Compute GRPO loss on verifiable reward
        grpo_loss = self._compute_loss(model, inputs)
        loss = grpo_loss

        if self.distill_mode == "student_topp_reverse_kl":
            self._metrics[mode]["reverse_kl/grpo_loss"].append(
                self.accelerator.gather(grpo_loss.detach()).nanmean().item()
            )

        if self.distill_mode in {"reverse_kl", "student_topp_reverse_kl"}:
            support_distill_loss = self._compute_support_distill_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=mask,
            )
            weighted_support_distill_loss = self.support_loss_coef * support_distill_loss

            if self.distill_mode == "reverse_kl":
                self._metrics[mode]["student_topp_reverse_kl/weighted_support_loss"].append(
                    self.accelerator.gather(weighted_support_distill_loss.detach()).nanmean().item()
                )
            elif self.distill_mode == "student_topp_reverse_kl":
                loss += weighted_support_distill_loss

        # Compute loss
        if self.single_step_decomposition:
            single_step_decomposition_loss = self._single_step_decomposition_loss(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                mask=mask,
            )

            loss += single_step_decomposition_loss

        # Empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = torch.zeros((), device=self.accelerator.device)
        return loss, None, None

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        if checkpoint_dir.exists():
            self._write_sample_rollout_file(checkpoint_dir)

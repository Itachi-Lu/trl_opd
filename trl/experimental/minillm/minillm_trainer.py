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
        self.support_loss_teacher_entropy_weighting = args.support_loss_teacher_entropy_weighting
        self.da_opd_weighting = args.da_opd_weighting
        self.da_opd_tau = args.da_opd_tau
        self.da_opd_normalization = args.da_opd_normalization
        self.da_opd_window_size = args.da_opd_window_size
        self.da_opd_ema_beta = args.da_opd_ema_beta
        self.opd_use_reward_advantage = args.opd_use_reward_advantage
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

    def _compute_da_opd_token_weights(
        self,
        teacher_token_logprobs: torch.Tensor,
        student_token_logprobs: torch.Tensor,
        mask: torch.Tensor,
        student_log_probs: torch.Tensor | None = None,
        teacher_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to a normalization-specific score function and convert to per-token weight.

        See `note/da_opd_theoretical_review.md` for the rationale of each option.

        Returns:
            weights: (B, T) detached weights, already masked.
            rho:     (B, T) raw cumulative log-prob ratio (always computed for monitoring).
        """
        mask_float = mask.float()
        logprob_ratio = (teacher_token_logprobs - student_token_logprobs) * mask_float
        rho = logprob_ratio.cumsum(dim=-1)

        norm = self.da_opd_normalization
        if norm == "raw":
            score = self._da_opd_score_raw(rho)
        elif norm == "seq":
            score = self._da_opd_score_seq(logprob_ratio, rho, mask_float)
        elif norm == "window_avg":
            score = self._da_opd_score_window_avg(rho, mask_float, self.da_opd_window_size)
        elif norm == "ema_kl":
            if student_log_probs is None or teacher_log_probs is None:
                raise RuntimeError(
                    "da_opd_normalization='ema_kl' requires student_log_probs and teacher_log_probs."
                )
            score = self._da_opd_score_ema_kl(student_log_probs, teacher_log_probs, mask_float, self.da_opd_ema_beta)
        elif norm == "inverse_length":
            score = self._da_opd_score_inverse_length(rho, mask_float)
        else:
            raise ValueError(f"Unsupported da_opd_normalization={norm!r}")

        weights = torch.sigmoid(score / self.da_opd_tau).detach() * mask_float
        return weights, rho

    @staticmethod
    def _da_opd_score_raw(rho: torch.Tensor) -> torch.Tensor:
        """Paper default: rho_prefix (cumulative log-prob ratio shifted by 1)."""
        return torch.cat([torch.zeros_like(rho[:, :1]), rho[:, :-1]], dim=-1)

    @staticmethod
    def _da_opd_score_seq(
        logprob_ratio: torch.Tensor, rho: torch.Tensor, mask_float: torch.Tensor
    ) -> torch.Tensor:
        """Narrative A: sequence-level scalar IS, per-token avg ratio broadcast across T."""
        rho_T = logprob_ratio.sum(dim=-1, keepdim=True)
        T_valid = mask_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        per_token_avg = rho_T / T_valid
        return per_token_avg.expand_as(rho)

    @staticmethod
    def _da_opd_score_window_avg(rho: torch.Tensor, mask_float: torch.Tensor, window: int) -> torch.Tensor:
        """Narrative B: average per-token log-prob ratio over a backward window of size `window`."""
        rho_prefix = torch.cat([torch.zeros_like(rho[:, :1]), rho[:, :-1]], dim=-1)
        # rho_shifted[t] = rho_prefix[t - window] (clamped to 0 for t < window)
        rho_shifted = torch.cat(
            [torch.zeros_like(rho_prefix[:, :window]), rho_prefix[:, :-window]], dim=-1
        )
        win_sum = rho_prefix - rho_shifted

        # Effective window length (number of valid prefix tokens within the last `window` positions)
        valid_prefix = torch.cat(
            [torch.zeros_like(mask_float[:, :1]), mask_float[:, :-1]], dim=-1
        ).cumsum(dim=-1)
        valid_shifted = torch.cat(
            [torch.zeros_like(valid_prefix[:, :window]), valid_prefix[:, :-window]], dim=-1
        )
        win_len = (valid_prefix - valid_shifted).clamp_min(1.0)
        return win_sum / win_len

    @staticmethod
    def _da_opd_score_inverse_length(rho: torch.Tensor, mask_float: torch.Tensor) -> torch.Tensor:
        """Narrative C: reverse direction. -rho_avg so that high cumulative drift -> high weight."""
        rho_prefix = torch.cat([torch.zeros_like(rho[:, :1]), rho[:, :-1]], dim=-1)
        valid_prefix = torch.cat(
            [torch.zeros_like(mask_float[:, :1]), mask_float[:, :-1]], dim=-1
        ).cumsum(dim=-1).clamp_min(1.0)
        return -(rho_prefix / valid_prefix)

    @staticmethod
    def _da_opd_score_ema_kl(
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        mask_float: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """Narrative B (alt): EMA of per-step full-distribution reverse KL.

        score_t = -EMA_{t-1}(KL(pi_theta || pi_te)).  Larger KL -> smaller sigmoid weight.
        Uses sequential scan; for T=18432 this is sub-second on GPU.
        """
        # Per-step KL(pi_theta || pi_te) using full vocab distributions.
        student_probs = student_log_probs.exp()
        kl_per_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        kl_per_token = kl_per_token * mask_float

        B, T = kl_per_token.shape
        ema = torch.zeros((B, T), dtype=kl_per_token.dtype, device=kl_per_token.device)
        prev = torch.zeros(B, dtype=kl_per_token.dtype, device=kl_per_token.device)
        one_minus = 1.0 - beta
        for t in range(T):
            prev = beta * prev + one_minus * kl_per_token[:, t]
            ema[:, t] = prev

        # Shift right so that score at position t uses EMA up to t-1 (causal).
        ema_prefix = torch.cat([torch.zeros_like(ema[:, :1]), ema[:, :-1]], dim=-1)
        return -ema_prefix

    def _log_da_opd_metrics(self, rho: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor, mode: str) -> None:
        valid_mask = mask.bool()
        mask_float = mask.float()
        rho_valid = rho.detach().float()[valid_mask]
        weight_valid = weights.detach().float()[valid_mask]

        if rho_valid.numel() == 0:
            rho_valid = torch.full((1,), float("nan"), device=rho.device)
        if weight_valid.numel() == 0:
            weight_valid = torch.full((1,), float("nan"), device=weights.device)

        rho_valid = self.accelerator.gather(
            self.accelerator.pad_across_processes(rho_valid, dim=0, pad_index=float("nan"))
        )
        weight_valid = self.accelerator.gather(
            self.accelerator.pad_across_processes(weight_valid, dim=0, pad_index=float("nan"))
        )
        rho_valid = rho_valid[~torch.isnan(rho_valid)]
        weight_valid = weight_valid[~torch.isnan(weight_valid)]

        def safe_stats(values: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if values.numel() == 0:
                zero = torch.zeros((), device=device)
                return zero, zero, zero
            return values.mean(), values.min(), values.max()

        rho_mean, rho_min, rho_max = safe_stats(rho_valid, rho.device)
        weight_mean, weight_min, weight_max = safe_stats(weight_valid, weights.device)
        self._metrics[mode]["da_opd/rho_mean"].append(rho_mean.item())
        self._metrics[mode]["da_opd/rho_min"].append(rho_min.item())
        self._metrics[mode]["da_opd/rho_max"].append(rho_max.item())
        self._metrics[mode]["da_opd/weight_mean"].append(weight_mean.item())
        self._metrics[mode]["da_opd/weight_min"].append(weight_min.item())
        self._metrics[mode]["da_opd/weight_max"].append(weight_max.item())

        # ---- New diagnostics introduced by da_opd_theoretical_review.md ----
        # Per-sample sums / max for effective_tokens / effective_fraction.
        weights_masked = weights.detach().float() * mask_float
        per_sample_weight_sum = weights_masked.sum(dim=-1)
        per_sample_weight_max = weights_masked.max(dim=-1).values.clamp_min(1e-8)
        per_sample_valid_count = mask_float.sum(dim=-1).clamp_min(1.0)

        effective_tokens = per_sample_weight_sum / per_sample_weight_max
        effective_fraction = per_sample_weight_sum / per_sample_valid_count

        # Saturation rates only over valid tokens.
        valid_count = mask_float.sum().clamp_min(1.0)
        sat_low = ((weights_masked < 1e-6) & valid_mask).float().sum() / valid_count
        sat_high = ((weights_masked > 0.99) & valid_mask).float().sum() / valid_count

        # Position-binned weight means (4 segments along the tensor T axis).
        T = weights.size(-1)
        seg_size = max(1, T // 4)

        def seg_mean(start: int, end: int) -> torch.Tensor:
            w_seg = weights_masked[:, start:end]
            m_seg = mask_float[:, start:end]
            return w_seg.sum() / m_seg.sum().clamp_min(1.0)

        seg0 = seg_mean(0, seg_size)
        seg1 = seg_mean(seg_size, 2 * seg_size)
        seg2 = seg_mean(2 * seg_size, 3 * seg_size)
        seg3 = seg_mean(3 * seg_size, T)

        # All scalar tensors; gather then nanmean for cross-rank aggregation.
        def gather_scalar(value: torch.Tensor) -> float:
            value = value.detach().to(self.accelerator.device)
            gathered = self.accelerator.gather(value.reshape(1))
            return gathered.float().nanmean().item()

        self._metrics[mode]["da_opd/effective_tokens"].append(gather_scalar(effective_tokens.mean()))
        self._metrics[mode]["da_opd/effective_fraction"].append(gather_scalar(effective_fraction.mean()))
        self._metrics[mode]["da_opd/saturation_rate_low"].append(gather_scalar(sat_low))
        self._metrics[mode]["da_opd/saturation_rate_high"].append(gather_scalar(sat_high))
        self._metrics[mode]["da_opd/weight_seg0_mean"].append(gather_scalar(seg0))
        self._metrics[mode]["da_opd/weight_seg1_mean"].append(gather_scalar(seg1))
        self._metrics[mode]["da_opd/weight_seg2_mean"].append(gather_scalar(seg2))
        self._metrics[mode]["da_opd/weight_seg3_mean"].append(gather_scalar(seg3))

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

    def _exclude_stop_tokens_from_support(
        self, support_indices: torch.Tensor, local_support_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.distill_mode != "student_topp_reverse_kl":
            return support_indices, local_support_mask

        excluded_token_ids = []
        if getattr(self, "pad_token_id", None) is not None:
            excluded_token_ids.append(self.pad_token_id)
        if getattr(self, "eos_token_id", None) is not None:
            excluded_token_ids.append(self.eos_token_id)
        if not excluded_token_ids:
            return support_indices, local_support_mask

        filtered_support_mask = local_support_mask.clone()
        for token_id in set(excluded_token_ids):
            filtered_support_mask &= support_indices != token_id

        support_width = support_indices.size(-1)
        rank_positions = torch.arange(support_width, device=support_indices.device).view(1, 1, support_width)
        reorder_keys = (~filtered_support_mask).long() * support_width + rank_positions
        reorder_indices = reorder_keys.argsort(dim=-1)
        filtered_support_indices = support_indices.gather(dim=-1, index=reorder_indices)

        filtered_support_sizes = filtered_support_mask.sum(dim=-1)
        filtered_local_support_mask = self._build_local_support_mask(
            filtered_support_sizes, support_width, support_indices.device
        )
        return filtered_support_indices, filtered_local_support_mask

    def _build_support_tensors(
        self,
        student_logits: torch.Tensor,
        top_p: float,
        min_k: int,
        exclude_stop_tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_indices, support_sizes = self._build_student_topp_support(student_logits, top_p=top_p, min_k=min_k)
        max_support_size = int(support_sizes.max().item())
        support_indices = topk_indices[..., :max_support_size]
        local_support_mask = self._build_local_support_mask(support_sizes, max_support_size, student_logits.device)
        if exclude_stop_tokens:
            support_indices, local_support_mask = self._exclude_stop_tokens_from_support(
                support_indices, local_support_mask
            )
        return support_indices, local_support_mask

    @staticmethod
    def _compute_teacher_mass(
        teacher_log_probs: torch.Tensor, support_indices: torch.Tensor, local_support_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        teacher_support_probs = teacher_log_probs.gather(dim=-1, index=support_indices).exp()
        if local_support_mask is not None:
            teacher_support_probs = teacher_support_probs * local_support_mask.to(teacher_support_probs.dtype)
        return teacher_support_probs.sum(dim=-1)

    @staticmethod
    def _map_normalized_entropy_to_weight(
        normalized_entropy: torch.Tensor, min_weight: float = 0.2, max_weight: float = 2.0
    ) -> torch.Tensor:
        normalized_entropy = normalized_entropy.clamp(min=0.0, max=1.0)
        return max_weight - (max_weight - min_weight) * normalized_entropy

    @staticmethod
    def _compute_teacher_full_vocab_entropy(teacher_log_probs: torch.Tensor) -> torch.Tensor:
        teacher_probs = teacher_log_probs.exp()
        teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
        vocab_size = teacher_log_probs.size(-1)
        if vocab_size > 1:
            teacher_entropy = teacher_entropy / math.log(vocab_size)
        else:
            teacher_entropy = torch.zeros_like(teacher_entropy)
        return teacher_entropy

    @staticmethod
    def _compute_teacher_support_entropy(
        teacher_log_probs: torch.Tensor,
        support_indices: torch.Tensor,
        local_support_mask: torch.Tensor,
        teacher_mass: torch.Tensor,
    ) -> torch.Tensor:
        teacher_support_probs = teacher_log_probs.gather(dim=-1, index=support_indices).exp()
        teacher_support_probs = teacher_support_probs * local_support_mask.to(teacher_support_probs.dtype)
        normalized_teacher_support_probs = teacher_support_probs / teacher_mass.unsqueeze(-1).clamp(min=1e-6)
        normalized_teacher_support_log_probs = torch.log(normalized_teacher_support_probs.clamp(min=1e-12))
        teacher_entropy = -(normalized_teacher_support_probs * normalized_teacher_support_log_probs).sum(dim=-1)

        support_sizes = local_support_mask.sum(dim=-1)
        support_entropy_denominator = torch.log(support_sizes.clamp(min=2).to(dtype=teacher_entropy.dtype))
        normalized_teacher_entropy = torch.zeros_like(teacher_entropy)
        valid_support = support_sizes > 1
        normalized_teacher_entropy = torch.where(
            valid_support,
            teacher_entropy / support_entropy_denominator.clamp(min=1e-6),
            normalized_teacher_entropy,
        )
        return normalized_teacher_entropy

    def _compute_teacher_entropy_weight(
        self,
        teacher_log_probs: torch.Tensor,
        support_indices: torch.Tensor,
        local_support_mask: torch.Tensor,
        teacher_mass: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.support_loss_teacher_entropy_weighting is None:
            return None

        if self.support_loss_teacher_entropy_weighting == "teacher_entropy_full_vocab":
            teacher_entropy = self._compute_teacher_full_vocab_entropy(teacher_log_probs)
            return self._map_normalized_entropy_to_weight(teacher_entropy)

        if self.support_loss_teacher_entropy_weighting == "teacher_entropy_support":
            teacher_entropy = self._compute_teacher_support_entropy(
                teacher_log_probs, support_indices, local_support_mask, teacher_mass
            )
            return self._map_normalized_entropy_to_weight(teacher_entropy)

        raise ValueError(
            "Unsupported support_loss_teacher_entropy_weighting: "
            f"{self.support_loss_teacher_entropy_weighting!r}."
        )

    def _log_masked_distribution_metrics(
        self, prefix: str, values: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> None:
        values = values.detach().float()
        valid_values = values[mask.bool()]

        # Completion tensors are padded only within each local batch, so their token dimension can differ across
        # ranks. Gather only the valid entries after flattening, then pad across processes to make the collective safe.
        if valid_values.numel() == 0:
            valid_values = torch.full((1,), float("nan"), device=values.device)
        padded_values = self.accelerator.pad_across_processes(valid_values, dim=0, pad_index=float("nan"))
        gathered_values = self.accelerator.gather(padded_values)
        valid_values = gathered_values[~torch.isnan(gathered_values)]

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

    def _log_masked_mean_metric(self, name: str, values: torch.Tensor, mask: torch.Tensor, mode: str) -> None:
        values = values.detach().float()
        valid_values = values[mask.bool()]

        if valid_values.numel() == 0:
            valid_values = torch.full((1,), float("nan"), device=values.device)
        padded_values = self.accelerator.pad_across_processes(valid_values, dim=0, pad_index=float("nan"))
        gathered_values = self.accelerator.gather(padded_values)
        valid_values = gathered_values[~torch.isnan(gathered_values)]

        if valid_values.numel() == 0:
            mean = torch.zeros((), device=values.device)
        else:
            mean = valid_values.mean()
        self._metrics[mode][f"{name}/mean"].append(mean.item())

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

            support_indices, local_support_mask = self._build_support_tensors(
                student_logits,
                top_p=self.support_top_p,
                min_k=self.support_min_k,
                exclude_stop_tokens=True,
            )
            teacher_mass = self._compute_teacher_mass(teacher_log_probs, support_indices, local_support_mask)
            self._log_masked_distribution_metrics("teacher_mass/support", teacher_mass, mask, mode)

    def _log_teacher_entropy_metrics(
        self, student_logits: torch.Tensor, teacher_log_probs: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> None:
        with torch.no_grad():
            mask_float = mask.float()
            valid_count = mask_float.sum().clamp(min=1.0)

            teacher_full_vocab_entropy = self._compute_teacher_full_vocab_entropy(teacher_log_probs)
            full_vocab_mean = (teacher_full_vocab_entropy * mask_float).sum() / valid_count
            self._metrics[mode]["teacher_entropy/full_vocab_mean"].append(
                self.accelerator.gather(full_vocab_mean).nanmean().item()
            )

            support_indices, local_support_mask = self._build_support_tensors(
                student_logits,
                top_p=self.support_top_p,
                min_k=self.support_min_k,
                exclude_stop_tokens=True,
            )
            teacher_mass = self._compute_teacher_mass(teacher_log_probs, support_indices, local_support_mask)
            teacher_support_entropy = self._compute_teacher_support_entropy(
                teacher_log_probs, support_indices, local_support_mask, teacher_mass
            )
            support_mean = (teacher_support_entropy * mask_float).sum() / valid_count
            self._metrics[mode]["teacher_entropy/support_mean"].append(
                self.accelerator.gather(support_mean).nanmean().item()
            )

    @staticmethod
    def _compute_topk_local_entropy(topk_log_probs: torch.Tensor) -> torch.Tensor:
        topk_probs = topk_log_probs.exp()
        normalized_topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        normalized_topk_log_probs = torch.log(normalized_topk_probs.clamp(min=1e-12))
        entropy = -(normalized_topk_probs * normalized_topk_log_probs).sum(dim=-1)
        topk_size = topk_log_probs.size(-1)
        if topk_size > 1:
            entropy = entropy / math.log(topk_size)
        else:
            entropy = torch.zeros_like(entropy)
        return entropy

    def _log_opd_alignment_metrics(
        self, student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> None:
        with torch.no_grad():
            max_topk = min(32, student_log_probs.size(-1))
            student_topk_log_probs, student_topk_indices = student_log_probs.detach().topk(max_topk, dim=-1)
            teacher_topk_log_probs, teacher_topk_indices = teacher_log_probs.detach().topk(max_topk, dim=-1)

            for k in (8, 16, 32):
                current_k = min(k, max_topk)
                student_indices = student_topk_indices[..., :current_k]
                teacher_indices = teacher_topk_indices[..., :current_k]
                student_log_probs_k = student_topk_log_probs[..., :current_k]
                teacher_log_probs_k = teacher_topk_log_probs[..., :current_k]

                student_overlap_mask = (student_indices.unsqueeze(-1) == teacher_indices.unsqueeze(-2)).any(dim=-1)
                teacher_overlap_mask = (teacher_indices.unsqueeze(-1) == student_indices.unsqueeze(-2)).any(dim=-1)

                overlap_ratio = student_overlap_mask.float().sum(dim=-1) / current_k
                student_overlap_mass = (student_log_probs_k.exp() * student_overlap_mask).sum(dim=-1)
                teacher_overlap_mass = (teacher_log_probs_k.exp() * teacher_overlap_mask).sum(dim=-1)

                student_entropy = self._compute_topk_local_entropy(student_log_probs_k)
                teacher_entropy = self._compute_topk_local_entropy(teacher_log_probs_k)
                entropy_gap = torch.abs(student_entropy - teacher_entropy)

                self._log_masked_mean_metric(f"opd/overlap_ratio/topk_{k}", overlap_ratio, mask, mode)
                self._log_masked_mean_metric(
                    f"opd/student_overlap_mass/topk_{k}", student_overlap_mass, mask, mode
                )
                self._log_masked_mean_metric(
                    f"opd/teacher_overlap_mass/topk_{k}", teacher_overlap_mass, mask, mode
                )
                self._log_masked_mean_metric(f"opd/entropy_gap/topk_{k}", entropy_gap, mask, mode)

    def _compute_support_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        support_indices, local_support_mask = self._build_support_tensors(
            student_logits,
            top_p=self.support_top_p,
            min_k=self.support_min_k,
            exclude_stop_tokens=True,
        )
        neg_large = torch.finfo(student_logits.dtype).min

        student_support_logits = student_logits.gather(dim=-1, index=support_indices)
        teacher_support_logits = teacher_logits.gather(dim=-1, index=support_indices)
        masked_student_support_logits = student_support_logits.masked_fill(~local_support_mask, neg_large)
        masked_teacher_support_logits = teacher_support_logits.masked_fill(~local_support_mask, neg_large)

        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_mass = self._compute_teacher_mass(teacher_log_probs, support_indices, local_support_mask)
        teacher_entropy_weight = self._compute_teacher_entropy_weight(
            teacher_log_probs, support_indices, local_support_mask, teacher_mass
        )

        local_student_log_probs = F.log_softmax(masked_student_support_logits, dim=-1)
        local_teacher_log_probs = F.log_softmax(masked_teacher_support_logits, dim=-1)
        local_student_probs = local_student_log_probs.exp()

        per_token_loss = (local_student_probs * (local_student_log_probs - local_teacher_log_probs)).sum(dim=-1)
        per_token_loss = per_token_loss * local_support_mask.any(dim=-1).to(per_token_loss.dtype)
        if self.support_loss_use_teacher_mass_weighting:
            per_token_loss = per_token_loss * teacher_mass
        if teacher_entropy_weight is not None:
            per_token_loss = per_token_loss * teacher_entropy_weight

        mask_float = mask.float()
        if token_weights is not None:
            reduction_weights = token_weights.to(dtype=per_token_loss.dtype) * mask_float
            return (per_token_loss * reduction_weights).sum() / reduction_weights.sum().clamp_min(1e-8)
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
        self._log_teacher_entropy_metrics(student_logits, teacher_log_probs, mask, mode)
        self._log_opd_alignment_metrics(student_log_probs, teacher_log_probs, mask, mode)

        if not self.opd_use_reward_advantage:
            inputs["advantages"] = torch.zeros_like(inputs["advantages"])

        da_opd_token_weights = None
        if self.da_opd_weighting:
            rollout_log_probs_on_labels = inputs.get("old_per_token_logps")
            if rollout_log_probs_on_labels is None:
                raise RuntimeError(
                    "DA-OPD weighting requires `old_per_token_logps`, but it was not found in the inputs."
                )
            da_opd_token_weights, da_opd_rho = self._compute_da_opd_token_weights(
                teacher_token_logprobs=teacher_log_probs_on_labels,
                student_token_logprobs=rollout_log_probs_on_labels,
                mask=mask,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
            )
            self._log_da_opd_metrics(da_opd_rho, da_opd_token_weights, mask, mode)

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
            if da_opd_token_weights is not None:
                inputs["token_loss_weights"] = da_opd_token_weights

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
                token_weights=da_opd_token_weights,
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

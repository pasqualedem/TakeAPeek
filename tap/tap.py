from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from tap.adapters import get_peft_config, get_peft_model
from tap.loss import FSSLoss
from tap.substitutor import Substitutor


class TakeAPeek:
    """
    Inference-time encoder adaptation for few-shot semantic segmentation.

    Wraps any encoder-decoder FSS model and applies LoRA-based support-set
    adaptation at inference time before predicting the query segmentation,
    as described in De Marinis et al., Pattern Recognition Letters, 2026.

    Args:
        model: FSS model whose ``forward(batch_dict)`` returns a dict with at
               least a ``"logits"`` key of shape ``(B, C, H, W)``.
        lora_config: A PEFT config object (e.g. ``LoraConfig``) or a plain dict
                     forwarded to ``get_peft_config("lora", ...)``.
        num_iterations: Number of outer adaptation iterations (T in the paper).
        lr: AdamW learning rate applied exclusively to the LoRA parameters.
        device: Device used for the adapted model and all tensors.
        substitutor: Optional custom :class:`~tap.substitutor.Substitutor`.
                     A default ``Substitutor()`` is created when omitted.

    Example (N-way K-shot)::

        from peft import LoraConfig
        from tap import TakeAPeek

        tap = TakeAPeek(
            model,
            LoraConfig(r=64, target_modules=["query", "value"]),
            num_iterations=8,
            lr=1e-3,
        )

        # batch["images"]: (B, M, C, H, W)
        #   M = 1 (query) + N*K (support images, N classes × K shots)
        # batch["classes"]: nested list encoding class identity per image
        # gt: (B, M, H, W) — query GT at index 0, support GTs at 1..M-1
        logits = tap(batch, gt)   # (B, C, H, W)
        pred   = logits.argmax(dim=1)
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config,
        *,
        num_iterations: int = 8,
        lr: float = 1e-3,
        device: str = "cuda",
        substitutor: Optional[Substitutor] = None,
    ) -> None:
        self.model = model
        self.lora_config = (
            get_peft_config("lora", lora_config)
            if isinstance(lora_config, dict)
            else lora_config
        )
        self.num_iterations = num_iterations
        self.lr = lr
        self.device = device
        self.substitutor = substitutor if substitutor is not None else Substitutor()
        self.loss = FSSLoss(
            class_weighting=True,
            components={"focal": {"weight": 1.0}},
        )

    def __call__(
        self,
        batch: dict,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adapt to the support set and return query segmentation logits.

        The LoRA parameters are re-initialised from scratch on every call so
        each episode is independent.

        Args:
            batch: Batch dict consumed by the FSS model.
                   ``batch["images"]`` has shape ``(B, M, C, H, W)`` where
                   ``M = 1 + N*K``: index 0 is the query image, indices
                   1…M-1 are the N*K support images (N classes, K shots each).
                   ``batch["classes"]`` encodes which class each support image
                   belongs to and is required for N-way (multiclass) episodes.
            gt: Ground-truth masks, shape ``(B, M, H, W)``.
                ``gt[:, 0]`` is the query GT — it is never used for
                optimisation and may be set to zeros if unavailable.
                ``gt[:, 1:]`` supervises the adaptation steps.

        Returns:
            Segmentation logits for the query image, shape ``(B, C, H, W)``.
        """
        lora_model, optimizer = self._init_lora()
        return self._adapt_and_predict(lora_model, optimizer, (batch, gt))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Normalisation layers that track running statistics must stay in eval
    # mode during adaptation to avoid corrupting them at inference time.
    _FROZEN_NORM_TYPES = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )

    def _init_lora(self) -> Tuple[nn.Module, AdamW]:
        """Fresh LoRA-wrapped copy of the base model with a matching optimizer."""
        lora_model = get_peft_model(deepcopy(self.model), self.lora_config)
        # Enable train mode so dropout / stochastic-depth are active during
        # adaptation (matching the paper's LoraEvaluator which never calls
        # model.eval()).  Norm layers with running stats stay in eval so their
        # tracked statistics are not corrupted at inference time.
        self._set_adapt_mode(lora_model)
        optimizer = AdamW(lora_model.parameters(), lr=self.lr)
        return lora_model.to(self.device), optimizer

    def _set_adapt_mode(self, model: nn.Module) -> None:
        """Train mode for all modules except normalisation layers with running stats."""
        model.train()
        for m in model.modules():
            if isinstance(m, self._FROZEN_NORM_TYPES):
                m.eval()

    def _adapt_and_predict(
        self,
        lora_model: nn.Module,
        optimizer: AdamW,
        batch_tuple: Tuple[dict, torch.Tensor],
    ) -> torch.Tensor:
        """
        Run ``num_iterations`` outer loops of support-set adaptation.

        Each outer loop the Substitutor yields M sub-batches:
          i=0   — query image forward pass in eval mode (deterministic,
                  no gradient); captures logits.
          i>0   — each support image treated as a pseudo-query; model in
                  train mode, backward + optimizer step on LoRA params only.
        """
        query_logits = None
        for _ in range(self.num_iterations):
            self.substitutor.reset(batch=batch_tuple)
            for i, (batch, step_gt) in enumerate(self.substitutor):
                optimizer.zero_grad()
                if i == 0:
                    lora_model.eval()
                    with torch.no_grad():
                        res = lora_model(batch)
                    query_logits = res["logits"].detach()
                    self._set_adapt_mode(lora_model)
                else:
                    res = lora_model(batch)
                    self.loss(res, step_gt).backward()
                    optimizer.step()
        return query_logits

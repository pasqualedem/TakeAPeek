import logging
import os
import wandb
import random
from einops import rearrange
import imageio
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import torchvision
from transformers import ViTMAEForPreTraining
from tqdm import tqdm

from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

import yaml
from tap.adapters import PEFT_CONFIGS, get_peft_config, get_peft_model
from tap.loss import FSSLoss
from tap.utils.metrics import (
    DistributedMulticlassJaccardIndex,
    to_global_multiclass,
)
from tap.data import get_dataloaders
from tap.models import model_registry
from tap.utils.utils import torch_dict_load, FakeTracker
from torch.optim import AdamW

import lovely_tensors as lt
import torch
import torch.nn as nn

from tap.substitutor import Substitutor, IncrementalSubstitutor
from tap.utils import (
    create_rgb_segmentation,
    print_trainable_parameters,
    random_foldername,
)

from accelerate import Accelerator

lt.monkey_patch()

substitutor_cls = {
    "default": Substitutor,
    "incremental": IncrementalSubstitutor,
}


COCO_PARAMS = {
            "name": "coco",
            "instances_path": "data/coco/annotations/instances_val2014.json",
            "img_dir": "data/coco/train_val_2017",
            "split": "val",
            "val_fold_idx": 3,
            "n_folds": 4,
            "n_shots": 1,
            "n_ways": 1,
            "do_subsample": False,
            "add_box_noise": False,
            "val_num_samples": 100,
}

PASCAL_PARAMS = {
    "name": "pascal",
    "data_dir": "data/pascal",
    "split": "val",
    "val_fold_idx": 3,
    "n_folds": 4,
    "n_shots": 1,
    "n_ways": 1,
    "do_subsample": False,
    "val_num_samples": 100,
    "ignore_borders": True,
}

DEEPGLOBE_PARAMS = {
    "name": "deepglobe",
    "datapath": "data",
    "split": "val",
    "val_fold_idx": 0,
    "n_shots": 2,
    "n_ways": 1,
    "val_num_samples": 100,
}

ISIC_PARAMS = {
    "name": "deepglobe",
    "datapath": "data/ISIC_cls",
    "split": "val",
    "val_fold_idx": 0,
    "n_shots": 2,
    "n_ways": 1,
    "val_num_samples": 100,
}

LUNG_PARAMS = {
    "name": "chest",
    "datapath": "data",
    "split": "test",
    "val_fold_idx": 0,
    "n_shots": 2,
    "n_ways": 1,
    "val_num_samples": 100,
}

COCO_NAME = "val_coco20i"
PASCAL_NAME = "val_pascal5i"
DEEPGLOBE_NAME = "val_deepglobe"
ISIC_NAME = "val_isic"
LUNG_NAME = "val_lung"

DATASETS = {
    "pascal": (PASCAL_NAME, PASCAL_PARAMS),
    "coco": (COCO_NAME, COCO_PARAMS),
    "deepglobe": (DEEPGLOBE_NAME, DEEPGLOBE_PARAMS),
    "isic": (ISIC_NAME, ISIC_PARAMS),
    'lung': (LUNG_NAME, LUNG_PARAMS),
}

dataset_args = {
    "datasets": {},
    "common": {
        "remove_small_annotations": True,
        "image_size": 480,
        "custom_preprocess": False,
    },
}

dataloader_args = {
    "num_workers": 0,
    "possible_batch_example_nums": [[1, 2, 4]],
    "val_possible_batch_example_nums": [[1, 1]],
    "prompt_types": ["mask"],
    "prompt_choice_level": ["episode"],
    "val_prompt_types": ["mask"],
}

la_params = {
    "class_attention": True,
    "example_class_attention": True,
    "example_attention": True,
    "class_encoder": {
        "bank_size": 100,
        "embed_dim": 256,
        "name": "RandomMatrixEncoder",
    },
    "embed_dim": 256,
    "fusion_transformer": "TwoWayTransformer",
    "image_embed_dim": 768,
    "image_size": 480,
    "spatial_convs": 3,
    "use_vit_sam_neck": False,
    "custom_preprocess": False,
}

def set_batchnorm_dropout_eval_mode(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
        if isinstance(module, (nn.SyncBatchNorm, SyncBatchNorm)):
            module.eval()
        if isinstance(module, (nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d)):
            module.eval()
        if isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            module.eval()
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.eval()
        if isinstance(module, nn.MultiheadAttention):
            module.eval()
        if isinstance(module, nn.LayerNorm):
            module.eval()
        # Stochastic depth or similar layers
        if hasattr(module, 'training') and hasattr(module, 'survival_prob'):
            module.training = False
            module.survival_prob = 1.0
        if hasattr(module, 'deterministic') and hasattr(module, 'training'):
            module.deterministic = True
            module.training = False

def get_la(dataset, val_fold_idx, **kwargs):
    name = "lam_mae_b"
    
    if dataset == "coco":
        path = {
            0: "checkpoints/la/mae256_fold0_sgqsatyi.safetensors",
            1: "checkpoints/la/mae256_fold1_p470zqp0.safetensors",
            2: "checkpoints/la/mae256_fold2_d2exzuiv.safetensors",
            3: "checkpoints/la/mae256_fold3_y04k97k7.safetensors",
        }[val_fold_idx]
    elif dataset == "pascal":
        path = {
            0: "checkpoints/la/pascal/model_fold0_u4ypi8a8.safetensors",
            1: "checkpoints/la/pascal/model_fold1_hufeobgi.safetensors",
            2: "checkpoints/la/pascal/model_fold2_djgh9s86.safetensors",
            3: "checkpoints/la/pascal/model_fold3_g31z1tm3.safetensors",
        }[val_fold_idx]
        la_params["class_attention"] = False
        la_params["example_class_attention"] = False

    
    image_size = 480
    model = model_registry[name](**la_params)
    weights = torch_dict_load(path)
    weights = {k[6:]: v for k, v in weights.items()}

    keys = model.load_state_dict(weights, strict=False)
    for key in keys.missing_keys:
        if key.startswith("image_encoder"):
            continue
        print(f"Missing key: {key}")
    for key in keys.unexpected_keys:
        print(f"Unexpected key: {key}")
    return model, image_size

def get_dcama(dataset, val_fold_idx, **kwargs):
    name = "dcama"
    params = dict(
        backbone_checkpoint="checkpoints/dcama/swin_base_patch4_window12_384.pth",
        model_checkpoint=f"checkpoints/dcama/{dataset}/swin_fold{val_fold_idx}.pt",
    )
    image_size = 384
    return model_registry[name](**params), image_size

def get_bam(dataset, k_shots, val_fold_idx, **kwargs):
    name = "bam"
    params = dict(
        shots=k_shots,
        val_fold_idx=val_fold_idx,
        dataset=dataset,
    )
    image_size = 641
    bam = model_registry[name](**params)
    set_batchnorm_dropout_eval_mode(bam)
    return bam, image_size

def get_hdmnet(k_shots, val_fold_idx, **kwargs):
    name = "hdmnet"
    params = dict(
        shots=k_shots,
        val_fold_idx=val_fold_idx,
    )
    image_size = 641
    hdmnet = model_registry[name](**params)
    set_batchnorm_dropout_eval_mode(hdmnet)
    return hdmnet, image_size


def get_dmtnet(k_shots, val_fold_idx, **kwargs):
    name = "dmtnet"
    params = dict(
        model_checkpoint="checkpoints/dmtnet.pt",
    )
    image_size = 400
    dmtnet = model_registry[name](**params)
    set_batchnorm_dropout_eval_mode(dmtnet)
    return dmtnet, image_size

def get_fptrans(dataset, val_fold_idx, k_shots, **kwargs):
    name = "fptrans"
    params = dict(
        val_fold_idx=val_fold_idx,
        k_shots=k_shots,
        dataset=dataset,
        backbone="ViT-B/16",
    )
    image_size = 480
    return model_registry[name](**params), image_size


def get_dcama_ada(dataset, val_fold_idx, n_ways, **kwargs):
    name = "dcama_ada"
    adapter_params = dict(
        adapter_weight=0.1,
        hidden_ratio=32,
        drop_ratio=0.4,
        momentum=0.99
    )
    
    params = dict(
        backbone_checkpoint="checkpoints/dcama/swin_base_patch4_window12_384.pth",
        model_checkpoint=f"checkpoints/dcama/{dataset}/swin_fold{val_fold_idx}.pt",
        fold=val_fold_idx,
        nways=n_ways,
        adapter_params=adapter_params
    )
    image_size = 384
    return model_registry[name](**params), image_size


def get_fptrans_ada(dataset, val_fold_idx, n_ways, k_shots, **kwargs):
    name = "fptrans_ada"
    adapter_params = dict(
        adapter_weight=0.1,
        hidden_ratio=32,
        drop_ratio=0.4,
        momentum=0.99
    )
    
    params = dict(
        val_fold_idx=val_fold_idx,
        n_ways=n_ways,
        k_shots=k_shots,
        dataset=dataset,
        backbone="ViT-B/16",
        adapter_params=adapter_params,
    )
    image_size = 480
    return model_registry[name](**params), image_size



def get_model(model_name, **kwargs):
    supported_models = {
        "label_anything": get_la,
        "dcama": get_dcama,
        "bam": get_bam,
        "hdmnet": get_hdmnet,
        "dmtnet": get_dmtnet,
        "fptrans": get_fptrans,
        "dcama_ada": get_dcama_ada,
        "fptrans_ada": get_fptrans_ada,
    }
    return supported_models[model_name](**kwargs)

class ViTModelWrapper(ViTMAEForPreTraining):
    def forward(self, x):
        h, w = x.shape[-2:]
        output = super().forward(x, interpolate_pos_encoding=True)
        hs = output.last_hidden_state[:, 1:, :]
        return rearrange(hs, "b (h w) c -> b c h w", h=h // 16).contiguous()

    def mae_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

class LoraEvaluator:
    def __init__(
        self,
        accelerator,
        model,
        dataloader,
        lora_config,
        num_iterations,
        lr,
        substitutor,
        print_folder,
        print_every=50,
        device="cuda",
        tracker=None,
        batch_size=1,
    ):
        self.accelerator = accelerator
        self.num_iterations = num_iterations
        self.lora_config = lora_config
        self.dataloader = dataloader
        self.print_every = print_every
        self.print_folder = print_folder
        self.device = device
        self.num_iterations = num_iterations
        self.lr = lr
        self.tracker = tracker

        os.makedirs(print_folder, exist_ok=True)
        self.dataset_categories = next(
            iter(dataloader.dataset.datasets.values())
        ).categories

        self.substitutor = substitutor
        self.model = model
        self.batch_size = batch_size

        temp_lora_model = get_peft_model(deepcopy(self.model), self.lora_config)
        print(f"Target modules: {temp_lora_model.targeted_module_names}")

        self.trainable_params = print_trainable_parameters(temp_lora_model)
        self.loss = FSSLoss(
            **{"class_weighting": True, "components": {"focal": {"weight": 1.0}}}
        )
        self.optimizer = AdamW(temp_lora_model.parameters(), lr=lr)
        self.mious = [
            DistributedMulticlassJaccardIndex(
                num_classes=len(self.dataset_categories) + 1,
                average="macro",
                ignore_index=-100,
            ).to(device)
            for _ in range(num_iterations)
        ]

        # Per-sample mIoU tracking (local, non-distributed)
        self.per_sample_mious = {k: [] for k in range(num_iterations)}
        self._sample_miou_fn = DistributedMulticlassJaccardIndex(
            num_classes=len(self.dataset_categories) + 1,
            average="macro",
            ignore_index=-100,
        ).to(device)

    def reset_lora(self):
        lora_model = get_peft_model(deepcopy(self.model), self.lora_config)
        optimizer = AdamW(lora_model.parameters(), lr=self.lr)
        lora_model = lora_model.to(self.device)
        return lora_model, optimizer

    def lora_step(self, lora_model, optimizer, batch_tuple, gt, bar):
        segmentation_preds = []
        for k in range(self.num_iterations):
            bar.set_description(
                f"Iteration {k} gpu memory: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
            )
            self.substitutor.reset(batch=batch_tuple)
            for i, (batch, gt) in enumerate(self.substitutor):
                optimizer.zero_grad()
                if i == 0:
                    with torch.no_grad():
                        res = lora_model(batch)
                        loss_value = self.loss(res, gt)
                else:
                    res = lora_model(batch)
                    loss_value = self.loss(res, gt)
                    loss_value.backward()
                    optimizer.step()
                preds = res["logits"].argmax(dim=1)
                glob_preds, glob_gt = to_global_multiclass(
                    batch["classes"], self.dataset_categories, preds, gt
                )
                if i == 0:
                    self.mious[k].update(glob_preds, glob_gt)
                    segmentation_preds.append(preds.detach().cpu())

                    # Per-sample mIoU (local)
                    with torch.no_grad():
                        sample_miou = self._sample_miou_fn(glob_preds, glob_gt).item()
                        self._sample_miou_fn.reset()
                    self.per_sample_mious[k].append(sample_miou)

        return segmentation_preds

    def print_results(self, i, batch_tuple, segmentation_preds):
        outfolder = f"{self.print_folder}/sample_{i}"
        os.makedirs(outfolder, exist_ok=True)
        segmentation_gts = [
            create_rgb_segmentation(batch_tuple[1][:, i].cpu())
            for i in range(batch_tuple[1].shape[1])
        ]
        segmentation_preds = [
            create_rgb_segmentation(pred) for pred in segmentation_preds
        ]
        resize_images = torchvision.transforms.functional.resize(
            batch_tuple[0]["images"][0], segmentation_gts[0].shape[2:]
        )
        plotted_images = torch.cat(
            [resize_images.cpu(), torch.cat(segmentation_gts)], dim=3
        )
        plotted_images.rgb.fig.savefig(f"{outfolder}/input_gt.png")
        for j, segmentation_pred in enumerate(segmentation_preds):
            segmentation_pred.rgb.fig.savefig(f"{outfolder}/pred_{j}.png")

        frame_duration = 0.5
        images = [
            imageio.imread(f"{outfolder}/pred_{i}.png")
            for i in range(self.num_iterations)
        ]
        imageio.mimsave(
            f"{outfolder}/segmentation.gif",
            images,
            duration=frame_duration * len(images),
        )

    def _compute_per_sample_metrics(self):
        """Win/loss analysis and iteration trend metrics from per-sample mIoU."""
        # Shape: (num_iterations, num_samples)
        all_iters = np.array([self.per_sample_mious[k] for k in range(self.num_iterations)])
        baseline  = all_iters[0]
        final     = all_iters[-1]
        deltas    = final - baseline

        # ── Win / loss ────────────────────────────────────────────────
        wins   = deltas > 0
        losses = deltas < 0
        degradation_threshold = 1.0  # mIoU points

        win_loss_metrics = {
            "per_sample/win_rate":           wins.mean(),
            "per_sample/loss_rate":          losses.mean(),
            "per_sample/mean_gain_winners":  deltas[wins].mean()   if wins.any()   else 0.0,
            "per_sample/mean_gain_losers":   deltas[losses].mean() if losses.any() else 0.0,
            "per_sample/degradation_rate":   (deltas < -degradation_threshold).mean(),
            "per_sample/mean_delta":         deltas.mean(),
            "per_sample/std_delta":          deltas.std(),
        }

        # ── Iteration trend ───────────────────────────────────────────
        # Shape: (num_iterations-1, num_samples)
        iter_deltas   = np.diff(all_iters, axis=0)
        signs         = np.sign(iter_deltas)
        sign_changes  = (np.diff(signs, axis=0) != 0).sum(axis=0)  # (num_samples,)
        positive_steps = (iter_deltas > 0).sum(axis=0)
        negative_steps = (iter_deltas < 0).sum(axis=0)
        # Trend score ∈ [-1, +1]: +1 = monotone increasing, 0 = oscillating, -1 = monotone decreasing
        trend_score   = signs.mean(axis=0)
        
        # At least one positive step rate: percentage of samples that had at least one positive step across iterations
        at_least_one_positive = (iter_deltas > 0).any(axis=0)  # (num_samples,)
        self.tracker.log({
            "per_sample/at_least_one_positive_rate": at_least_one_positive.mean()
        })
        
        # Never positive rate: percentage of samples that had no positive steps across iterations
        never_positive = (~at_least_one_positive).mean()
        self.tracker.log({
            "per_sample/never_positive_rate": never_positive
        })

        iter_trend_metrics = {
            "iter_trend/mean_positive_steps":    positive_steps.mean(),
            "iter_trend/mean_negative_steps":    negative_steps.mean(),
            "iter_trend/std_positive_steps":     positive_steps.std(),
            "iter_trend/positive_step_rate":     (iter_deltas > 0).mean(),
            "iter_trend/monotone_increasing":    (positive_steps == self.num_iterations - 1).mean(),
            "iter_trend/monotone_decreasing":    (negative_steps == self.num_iterations - 1).mean(),
            "iter_trend/mean_sign_changes":      sign_changes.mean(),
            "iter_trend/std_sign_changes":       sign_changes.std(),
            "iter_trend/monotone_rate":          (sign_changes == 0).mean(),
            "iter_trend/max_oscillation_rate":   (sign_changes == self.num_iterations - 2).mean(),
            "iter_trend/mean_trend_score":       trend_score.mean(),
            "iter_trend/std_trend_score":        trend_score.std(),
        }
        
        # for each iteration, compute the positive step rate and negative step rate
        for k in range(self.num_iterations - 1):
            iter_trend_metrics[f"iter_trend/positive_step_rate_it_{k}"] = (iter_deltas[k] > 0).mean()
            iter_trend_metrics[f"iter_trend/negative_step_rate_it_{k}"] = (iter_deltas[k] < 0).mean()

        return {**win_loss_metrics, **iter_trend_metrics}, all_iters, deltas, iter_deltas, sign_changes, trend_score

    def _plot_mious(self, miou_values, all_iters, deltas, iter_deltas, sign_changes, trend_score):
        baseline = all_iters[0]
        final    = all_iters[-1]
        sort_idx = np.argsort(final - baseline)

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle("mIoU Analysis", fontsize=13, fontweight="bold")

        # ── 1. Aggregate mIoU over iterations ────────────────────────
        axes[0, 0].plot(miou_values, marker="o", lw=1.8, color="#1a6fa8")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("mIoU (aggregate)")
        axes[0, 0].set_title("Aggregate mIoU")
        axes[0, 0].grid(True, linestyle=":", linewidth=0.6)

        # ── 2. Per-sample delta distribution ─────────────────────────
        axes[0, 1].hist(deltas, bins=30, color="#1a6fa8", edgecolor="white", linewidth=0.4)
        axes[0, 1].axvline(0, color="#c0392b", lw=1.2, ls="--")
        axes[0, 1].set_xlabel("ΔmIoU (final − baseline)")
        axes[0, 1].set_ylabel("# samples")
        axes[0, 1].set_title("Per-sample gain distribution")
        axes[0, 1].grid(True, linestyle=":", linewidth=0.6)

        # ── 3. Trend heatmap: positive step per iteration ─────────────
        im = axes[1, 0].imshow(
            (iter_deltas[:, sort_idx] > 0).astype(float),
            aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
            interpolation="nearest"
        )
        axes[1, 0].set_xlabel("Samples (sorted by final gain)")
        axes[1, 0].set_ylabel("Iteration step")
        axes[1, 0].set_yticks(range(self.num_iterations - 1))
        axes[1, 0].set_yticklabels(
            [f"{k}→{k+1}" for k in range(self.num_iterations - 1)], fontsize=7
        )
        axes[1, 0].set_title("Step-wise improvement heatmap")
        plt.colorbar(im, ax=axes[1, 0], label="positive step", shrink=0.8)

        # ── 4. Oscillation: sign changes vs trend score ───────────────
        sc = axes[1, 1].scatter(
            trend_score, sign_changes,
            c=deltas, cmap="RdYlGn", alpha=0.6, s=18,
            vmin=np.percentile(deltas, 5), vmax=np.percentile(deltas, 95)
        )
        axes[1, 1].set_xlabel("Trend score  (−1 = decreasing, +1 = increasing)")
        axes[1, 1].set_ylabel("Sign changes (oscillation)")
        axes[1, 1].set_title("Trend score vs Oscillation")
        axes[1, 1].grid(True, linestyle=":", linewidth=0.6)
        plt.colorbar(sc, ax=axes[1, 1], label="ΔmIoU", shrink=0.8)

        plt.tight_layout()
        plt.savefig(f"{self.print_folder}/mious.png", dpi=150)
        plt.close()

    def print_mious(self):
        print("Printing mious")
        miou_values = [miou.compute().item() for miou in self.mious]

        # ── Aggregate logging ─────────────────────────────────────────
        for i, miou_value in enumerate(miou_values):
            if i == 0:
                self.tracker.log({"miou_orig": miou_value})
            else:
                self.tracker.log({f"miou_it_{i}": miou_value})
                self.tracker.log({f"gain_it_{i}": miou_value - miou_values[0]})
                self.tracker.log({"miou": miou_value})
                self.tracker.log({"gain": miou_value - miou_values[0]})
            print(f"Iteration {i}: miou: {miou_value}")

        best_miou = max(miou_values)
        self.tracker.log({"best_miou": best_miou})
        self.tracker.log({"best_gain": best_miou - miou_values[0]})

        # ── Per-sample & trend metrics ────────────────────────────────
        metrics, all_iters, deltas, iter_deltas, sign_changes, trend_score = \
            self._compute_per_sample_metrics()

        for k, v in metrics.items():
            self.tracker.log({k: v})
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        # ── Plots ─────────────────────────────────────────────────────
        self._plot_mious(miou_values, all_iters, deltas, iter_deltas, sign_changes, trend_score)

        with open(f"{self.print_folder}/mious.txt", "w") as f:
            f.write("\n".join([str(m) for m in miou_values]))

    def split_batch(self, batch_tuple, num_splits):
        micro_batches = []
        batch, gt = batch_tuple
        for i in range(num_splits):
            micro_batch = {
                k: v[i : i + 1] if v is not None else None for k, v in batch.items()
            }
            micro_batch_tuple = (micro_batch, gt[i : i + 1])
            micro_batches.append(micro_batch_tuple)
        return micro_batches

    def evaluate(self):
        streams = [torch.cuda.Stream() for _ in range(self.batch_size)]
        bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))

        for i, (batch_tuple, data_name) in bar:
            micro_batches = self.split_batch(batch_tuple, self.batch_size)
            preds = [None] * len(micro_batches)

            for j, micro_batch in enumerate(micro_batches):
                with torch.cuda.stream(streams[j]):
                    lora_model, optimizer = self.reset_lora()
                    preds[j] = self.lora_step(
                        lora_model=lora_model,
                        optimizer=optimizer,
                        batch_tuple=micro_batch,
                        gt=micro_batch[1],
                        bar=bar,
                    )

            torch.cuda.synchronize()

            for j in range(len(micro_batches)):
                if (i * self.batch_size + j) % self.print_every == 0:
                    self.print_results(i * self.batch_size + j, micro_batches[j], preds[j])

            bar.set_postfix(
                miou0=self.mious[0].compute().item(),
                miouN=self.mious[-1].compute().item(),
            )

        self.print_mious()

def main(params):
    foldername = random_foldername()

    # Set all the seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Numpy random seed
    np.random.seed(seed)
    # Python random seed
    random.seed(seed)

    num_iterations = params.get("num_iterations", 10)
    device = params.get("device", "cuda")
    lora_r = params.get("lora_r", 32)
    lora_alpha = params.get("lora_alpha", None) or float(lora_r)
    lr = params.get("lr", 1e-4)
    target_modules = params.get("target_modules", ["query", "value"])
    lora_dropout = params.get("lora_dropout", 0.1)
    substitutor = params.get("substitutor", "default")
    subsample = params.get("subsample", None)
    augment = params.get("augment", False)
    n_ways = params.get("n_ways", 2)
    k_shots = params.get("k_shots", 5)
    val_num_samples = params.get("val_num_samples", 100)
    model_name = params.get("model", "tap")
    val_fold_idx = params.get("val_fold_idx", 3)
    dataset = params.get("dataset", "coco")
    mask_perturbation = params.get("mask_perturbation", 0.0)
    batch_size = params.get("batch_size", 1)

    # Initialize Accelerator
    accelerator = Accelerator()

    model, image_size = get_model(
        model_name, dataset=dataset, n_ways=n_ways, k_shots=k_shots, val_fold_idx=val_fold_idx
    )
    model = accelerator.prepare(model)
    
    dataset_name, datasets_params = DATASETS[dataset]
    dataset_args["datasets"][dataset_name] = datasets_params

    dataset_args["datasets"][dataset_name]["n_ways"] = n_ways
    dataset_args["datasets"][dataset_name]["n_shots"] = k_shots
    dataset_args["datasets"][dataset_name]["val_num_samples"] = val_num_samples
    dataset_args["datasets"][dataset_name]["val_fold_idx"] = val_fold_idx
    dataset_args["mask_perturbation"] = mask_perturbation
    dataset_args["common"]["image_size"] = image_size
    dataloader_args["val_possible_batch_example_nums"] = [[batch_size, 1]]

    val_dict = get_dataloaders(
        dataset_args, dataloader_args, num_processes=1
    )
    val = val_dict[dataset_name]
    val = accelerator.prepare(val)
    
    peft_params = dict(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    lora_config = get_peft_config(params.get("peft_type", "lora"), peft_params)

    folder = "offline"
    os.makedirs(folder, exist_ok=True)
    folder = "offline/lora"
    os.makedirs(folder, exist_ok=True)
    # Create a subfolder with the current time
    subfolder = f"{folder}/{foldername}"
    os.makedirs(subfolder, exist_ok=True)

    # Print params as yaml
    with open(f"{subfolder}/params.yaml", "w") as f:
        yaml.dump(params, f)

    if accelerator.is_main_process:
        cache_dir = "tmp"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["WANDB_ARTIFACT_LOCATION"] = cache_dir
        os.environ["WANDB_ARTIFACT_DIR"] = cache_dir
        os.environ["WANDB_CACHE_DIR"] = cache_dir
        os.environ["WANDB_CONFIG_DIR"] = cache_dir
        os.environ["WANDB_DATA_DIR"] = cache_dir
        group = params.get("experiment", {}).get("group", None)
        tracker = wandb.init(project="lorafss", config=params, group=group)
    else:
        tracker = FakeTracker()

    substitutor = substitutor_cls[substitutor](
        substitute=True,
        long_side_length=480,
        custom_preprocess=False,
        n_ways=n_ways,
        k_shots=k_shots,
        subsample=subsample,
        augment=augment,
    )

    lora_evaluator = LoraEvaluator(
        accelerator,
        model,
        val,
        lora_config,
        num_iterations,
        lr,
        print_folder=subfolder,
        device=device,
        tracker=tracker,
        substitutor=substitutor,
        batch_size=batch_size,
    )
    tracker.log({"trainable_params": lora_evaluator.trainable_params})

    lora_evaluator.evaluate()
    tracker.finish()

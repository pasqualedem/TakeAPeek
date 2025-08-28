import torch
import torch.nn.functional as F

from einops import rearrange, repeat

from tap.models.dcama_ada.dcama import DCAMA_AdaptiveFSS
from tap.utils.utils import ResultDict
from tap.data.utils import BatchKeys
from tap.data.utils import get_preprocess_shape


def build_dcama_ada(
    backbone: str = "swin",
    backbone_checkpoint: str = "checkpoints/backbone.pth",
    model_checkpoint: str = "checkpoints/dcama.pth",
    image_size: int = 384,
    benchmark: str = None,
    fold: int = None,
    adapter_params: dict = {},
    nshot: int = 1,
    custom_preprocess: bool = False,
):
    model = DCAMAMultiClass_Ada(
        backbone=backbone,
        backbone_checkpoint=backbone_checkpoint,
        benchmark=benchmark,
        fold=fold,
        adapter_params=adapter_params,
    )
    params = model.state_dict()
    params = {k: v for k, v in params.items() if "adapter" not in k}
    state_dict = torch.load(model_checkpoint, map_location="cpu")

    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert not unexpected_keys, f"Unexpected keys found in backbone checkpoint: {unexpected_keys}"
    assert all("adapter" in k for k in missing_keys), f"Missing adapter keys in backbone checkpoint: {missing_keys}"
    return model


class DCAMAMultiClass_Ada(DCAMA_AdaptiveFSS):
    def __init__(
        self, backbone, backbone_checkpoint, benchmark, fold, adapter_params
    ):
        self.predict = None
        self.generate_class_embeddings = None
        super().__init__(backbone, backbone_checkpoint, benchmark, fold, adapter_params)

    def _preprocess_masks(self, masks, dims):
        B, N, C, H, W = masks.size()
        # remove bg from masks
        masks = masks[:, :, 1:, ::]
        mask_size = 256

        # Repeat dims along class dimension
        support_dims = dims[:, 1:]
        repeated_dims = repeat(support_dims, "b n d -> (b n c) d", c=C)
        masks = rearrange(masks, "b n c h w -> (b n c) h w")

        # Remove padding from masks
        # pad_dims = [get_preprocess_shape(h, w, mask_size) for h, w in repeated_dims]
        # masks = [mask[:h, :w] for mask, (h, w) in zip(masks, pad_dims)]
        # masks = torch.cat(
        #     [
        #         F.interpolate(
        #             torch.unsqueeze(mask, 0).unsqueeze(0),
        #             size=(self.image_size, self.image_size),
        #             mode="nearest",
        #         )[0]
        #         for mask in masks
        #     ]
        # )
        return rearrange(masks, "(b n c) h w -> b n c h w", b=B, n=N)

    def forward(self, x, return_query_feats=False, return_support_feats=False):
        query_feats = None
        support_feats = []

        x[BatchKeys.PROMPT_MASKS] = self._preprocess_masks(
            x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS]
        )
        assert (
            x[BatchKeys.PROMPT_MASKS].shape[0] == 1
        ), "Only tested with batch size = 1"
        logits = []
        query = x[BatchKeys.IMAGES][:, :1]
        support = x[BatchKeys.IMAGES][:, 1:]
        # get logits for each class
        for c in range(x[BatchKeys.PROMPT_MASKS].size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            class_idx = [c for _ in x["classes"]]
            class_input_dict = {
                BatchKeys.IMAGES: torch.cat(
                    [query, support[class_examples].unsqueeze(0)], dim=1
                ),
                BatchKeys.PROMPT_MASKS: x[BatchKeys.PROMPT_MASKS][:, :, c, ::][
                    class_examples
                ].unsqueeze(0),
            }
            if self.training:
                class_logits = super().forward(class_input_dict, class_idx=class_idx)
            else:
                class_logits = super().forward_5shot_test(class_input_dict, class_idx=class_idx)
            logits.append(class_logits)
        logits = torch.stack(logits, dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = self.postprocess_masks(logits, x["dims"])

        return {
            ResultDict.LOGITS: logits,
            ResultDict.QUERY_FEATS: query_feats,
            ResultDict.SUPPORT_FEATS: support_feats,
        }

    def postprocess_masks(self, logits, dims):
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        dims = dims[:, 0, :]  # get real sizes of the query images
        logits = [
            F.interpolate(
                torch.unsqueeze(logit, 0),
                size=dim.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for logit, dim in zip(logits, dims)
        ]

        logits = torch.cat(
            [
                F.pad(
                    mask,
                    (
                        0,
                        max_dims[1] - dims[i, 1],
                        0,
                        max_dims[0] - dims[i, 0],
                    ),
                    mode="constant",
                    value=float("-inf"),
                )
                for i, mask in enumerate(logits)
            ]
        )
        return logits

    def get_learnable_params(self, train_params):
        return self.parameters()

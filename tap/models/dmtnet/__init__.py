import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from tap.data.utils import BatchKeys
from tap.models.dmtnet.dmtnet import DMTNetwork
from tap.utils.utils import ResultDict


def build_dmtnet(backbone="resnet50", model_checkpoint=None):
    model = DMTNetMultiClass(backbone)
    src_dict = torch.load(model_checkpoint, map_location="cpu")
    src_dict = {k[len("module."):]: v for k, v in src_dict.items()}
    model.load_state_dict(src_dict)
    return model


class DMTNetMultiClass(DMTNetwork):
    def __init__(self, *args, **kwargs):
        self.predict = None
        self.generate_class_embeddings = None
        super().__init__(*args, **kwargs)

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

    def forward(self, x):

        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS])
        assert masks.shape[0] == 1, "Only tested with batch size = 1"
        voting_masks = []
        fg_logits_masks = []
        # get logits for each class
        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = class_examples.sum().item()
            class_input_dict = {
                "query_img": x[BatchKeys.IMAGES][:, 0],
                "support_imgs": x[BatchKeys.IMAGES][:, 1:][class_examples].unsqueeze(0),
                "support_masks": masks[:, :, c, ::][class_examples].unsqueeze(0),
            }
            if n_shots == 1:
                logit_mask, bg_logit_mask, pred_mask = self.predict_mask_1shot(
                    class_input_dict["query_img"],
                    class_input_dict["support_imgs"][:, 0],
                    class_input_dict["support_masks"][:, 0],
                )
                fg_logits_masks.append(logit_mask)
            else:
                (voting_mask, logit_mask_orig, bg_logit_mask_orig) = (
                    self.predict_mask_nshot(class_input_dict, n_shots)
                )
                voting_masks.append(voting_mask)
                
        if fg_logits_masks:
            raw_logits = torch.stack(fg_logits_masks, dim=1)
            raw_logits = F.softmax(raw_logits, dim=2)
            fg_logits = raw_logits[:, :, 1, ::]
            bg_logits = raw_logits[:, :, 0, ::]
            bg_positions = fg_logits.argmax(dim=1)
            bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
            logits = torch.cat([bg_logits, fg_logits], dim=1)
        else:
            votes = torch.stack([class_res for class_res in voting_masks], dim=1)
            preds = (votes.argmax(dim=1)+1) * (votes > 0.5).max(dim=1).values
            logits = rearrange(F.one_hot(preds, num_classes=len(voting_masks)+1), "b h w c -> b c h w").float()
            
        logits = self.postprocess_masks(logits, x["dims"])

        return {
            ResultDict.LOGITS: logits,
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
        # set padding to background class
        logits[:, 0, :, :][logits[:, 0, :, :] == float("-inf")] = 0
        return logits

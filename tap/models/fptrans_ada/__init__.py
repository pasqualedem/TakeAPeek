import os
import gdown
import logging

import torch
import torch.nn.functional as F

from einops import rearrange, repeat

from .FPTrans import FPTrans_AdaptiveFSS
from tap.data.utils import BatchKeys
from tap.utils.utils import ResultDict


__networks = {
    "fptrans": FPTrans,
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_model(opt, logger, *args, **kwargs):
    if opt.network.lower() in __networks:
        model = __networks[opt.network.lower()](opt, logger, *args, **kwargs)
        if opt.print_model:
            print(model)
        return model
    else:
        raise ValueError(
            f"Not supported network: {opt.network}. {list(__networks.keys())}"
        )


fp_trans_dict = {
    "pascal": {
        "ViT-B/16": {
            1: [1, 2, 3, 4],
            5: [17, 18, 19, 20],
        },
        "DeiT-B/16": {
            1: [5, 6, 7, 8],
            5: [21, 22, 23, 24],
        },
        "DeiT-S/16": {
            1: [9, 10, 11, 12],
        },
        "DeiT-T/16": {
            1: [13, 14, 15, 16],
        },
    },
    "coco": {
        "ViT-B/16": {
            1: [25, 26, 27, 28],
            5: [33, 34, 35, 36],
        },
        "DeiT-B/16": {
            1: [29, 30, 31, 32],
            5: [37, 38, 39, 40],
        },
    },
}

fp_trans_experiment_links = [
    "1rvO04d4aK4m1zk-LuU20vSibBgHzGF8k",
    "1p3lQZ79ZHYwco4BsqWINEMjNrSuSxiaI",
    "1pcoi_mX_ZV0VPwSbtEiAXFejPH2AkiSs",
    "1_DzwW_mt7RnL8vw2FQc0k4wJcllJ5NXV",
    "1cQRPeYlM08uah8IipuaL5jvy6JHSGulX",
    "1D66YBSwo3aVZoRuk4nZBtK7Orgf_t2Dp",
    "1GetCnMR35x72HHdQX3rnZ9JE0XFttYfB",
    "1BNIUHWWvKr1Wueq2y5DkIuwbBuoshNct",
    "1nU6d7Te_dLxAs9cXzf5968pWIGmyqVSg",
    "1_S54vYuk_-Q__BnIBTmbYROH6T3Zpw6o",
    "1UXqpduUJfqVRG4OC6r6MSXS1iXvqpjK5",
    "17DjVlaJ4o8gJVTugDCR2IjhRgSX1FFsN",
    "1O74FJiQXxbU1hFms1VC7Xu3BzZsrjGsk",
    "1SyCBt3pjQZ-pxhbUzac2n9ICX6ZbGSeB",
    "16AKosExMK1U6F4gD-8Tuvw4QcHkAy_dH",
    "1DZyAwJTsGlC5IAe-QNYO1Vb8KcaHAkgX",
    "1l3-5oSyCIwF5FAaTpeyVcwP7h3Fzj1eZ",
    "1-JFQ6kY8FA5iIlBOItWy8UW_lVwlZ_xx",
    "1TjXp3d5EMb_divw6I3zsgomQKvaLxsRZ",
    "10-GqdhSe-zUpZo_7qLid8UAu1AQHToxX",
    "1T4Y0ykABYa8oblVpgESV0F_uPC3iOyoW",
    "1M2qxpJbZ2d9xUBXVDJ15pioGpzzEyjgo",
    "1owxRSd92CK9FvCCd6Ty9zDhqRG-5Oc8Q",
    "13wpVNYgc0-tNZzMpnCGge7v1UypJUDG-",
    "1cqh1mt7sETkZlQgi_lGDiT57MmDIke2c",
    "19TkTsFI74E5wm-cdzQfNHxduS3U57VsZ",
    "1C_psluZWRjUCU1WrUNINa97gN9UoNtph",
    "1IAdjuFA3T4MkeTRzJb2V0zpW3pdbtZhY",
    "1DXbwFrs12t0DuDYsB7qR_4Zr7O2b6CaL",
    "1201WaBHEaykqWI_3CXbkHU-GlkiY5HQd",
    "1asiKKhOd7YFogkqdM8T5mlSc_Db_QABJ",
    "1kQaxNUWDtpNK2euF9u9Az3W4A-4Tcp_V",
    "1A1YmjAk8fcCLtZq8lBD_JMAWFQahsv8Z",
    "1bS5gXHWXM3unjkKuq2VE8Hgjy6bHC4oR",
    "14WVbJjXG6M5wQxQ8r9vpqhB3KVKSLIug",
    "1H4P5ePnO9LivvrA4M51WC1AW5ENq1V99",
    "1Oa-6G5DhPX19lVMn3IU7Xd5rQWq-NFOt",
    "1ERsUZ842ENPQdpuWIEITZFxbwPV2AEQH",
    "1cfVM1mBgL9eHfzwh6-2bg2lN_QZTBtJW",
    "14WqeDmdqEKRGt9HrkcO4Nyt_feO3f1vo",
]


def build_fptrans_ada(
    model_checkpoint: str = None,
    dataset: str = "coco",  # can be "coco" or "pascal"
    backbone: str = "ViT-B/16",
    val_fold_idx: int = None,
    n_ways: int = 1,
    image_size: int = 480,
    k_shots: int = 5,
    adapter_params: dict = {},
):

    if k_shots not in [1, 5]:
        rounded_k = 1 if k_shots < 3 else 5
        print(f"Warning: k_shots should be either 1 or 5. Rounding to {rounded_k}.")
        k_shots = rounded_k

    bg_num = 5
    opt = {
        "nways": n_ways,
        "shot": k_shots,
        "image_size": image_size,
        "drop_dim": 1,
        "drop_rate": 0.1,
        "block_size": 4,
        "backbone": "ViT-B/16-384",
        "tqdm": False,
        "height": 480,
        "bg_num": bg_num,
        "num_prompt": 12 * (1 + bg_num * k_shots),
        "vit_stride": None,
        "dataset": dataset.upper(),
        "coco2pascal": False,
        "pt_std": 0.02,
        "vit_depth": 10,
    }
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if dataset is not None and val_fold_idx is not None and k_shots is not None:
        experiment_id = fp_trans_dict[dataset][backbone][k_shots][val_fold_idx]
        model_checkpoint = f"checkpoints/fptrans/{experiment_id}.pth"
        if os.path.exists(model_checkpoint):
            print(
                f"Using original checkpoint from experiment: {experiment_id} with val_fold_idx {val_fold_idx}, k_shots {k_shots} for dataset {dataset} and backbone {backbone}"
            )
        else:
            print(
                f"Downloading checkpoint for experiment: {experiment_id} with val_fold_idx {val_fold_idx}, k_shots {k_shots} for dataset {dataset} and backbone {backbone}"
            )
            if not os.path.exists("checkpoints/fptrans"):
                os.makedirs("checkpoints/fptrans")
            gdown.download(
                id=fp_trans_experiment_links[experiment_id - 1],
                output=model_checkpoint,
                quiet=False,
            )

    opt = dotdict(opt)
    model = FPTransAdaMultiClass(opt, adapter_params=adapter_params)
    model.load_weights(model_checkpoint, logger)

    return model


class FPTransAdaMultiClass(FPTrans_AdaptiveFSS):
    def _preprocess_masks(self, masks, H, W):
        masks = F.interpolate(masks, size=(masks.shape[2], H, W), mode="nearest")  # B, M, C, H, W

        B, N, C, _, _ = masks.size()
        # remove bg from masks
        masks = masks[:, :, 1:, ::]

        # Repeat dims along class dimension
        masks = rearrange(masks, "b n c h w -> (b n c) h w")

        return rearrange(masks, "(b n c) h w -> b n c h w", b=B, n=N)

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

    def forward(self, x):
        B, M, Ch, H, W = x[BatchKeys.IMAGES].size()
        S = M - 1

        q = x[BatchKeys.IMAGES][:, :1]
        s_x = x[BatchKeys.IMAGES][:, 1:]

        s_y = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS], H, W)  # B, S, C, H, W
        C = s_y.size(2)

        logits = []

        for c in range(C):
            class_idx = [c for _ in x["classes"]] if self.training else None
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            class_s_x = s_x[class_examples].unsqueeze(0)
            class_s_y = s_y[:, :, c, ::][class_examples].unsqueeze(0)
            logits.append(super().forward(q, class_s_x, class_s_y, None, None, class_idx=class_idx))

        logits = torch.stack([l["out"] for l in logits], dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = self.postprocess_masks(logits, x["dims"])
        return {ResultDict.LOGITS: logits}

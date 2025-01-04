import os
from easydict import EasyDict

import torch
import torch.nn.functional as F

from tap.data.utils import BatchKeys
from tap.utils.utils import ResultDict
from .HDMNet import OneModel


class HDMNetModel(OneModel):
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
    
    def forward(self, batch: dict):
        # Remove background from masks
        masks = batch[BatchKeys.PROMPT_MASKS][:, :, 1:]
        y_m, y_b, cat_idx = None, None, None
        logits = []

        # Iterate over each class to compute logits
        for c in range(masks.size(2)):
            # Extract class-specific examples and data
            class_examples = batch[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            x = batch[BatchKeys.IMAGES][:, 0]
            s_x = batch[BatchKeys.IMAGES][:, 1:][class_examples].unsqueeze(0)
            s_y = masks[:, :, c][class_examples].unsqueeze(0)
            
            # Count the number of shots
            n_shots = class_examples.sum().item()
            
            # Handle fewer shots than required
            if n_shots < self.shot:
                s_x = torch.cat(
                    [s_x, s_x[:, -1].unsqueeze(0).repeat(1, self.shot - n_shots, 1, 1, 1)],
                    dim=1
                )
                s_y = torch.cat(
                    [s_y, s_y[:, -1].unsqueeze(0).repeat(1, self.shot - n_shots, 1, 1)],
                    dim=1
                )
            
            # Append the logits computed for this class
            class_logits = super().forward(x, s_x=s_x, s_y=s_y, y_m=y_m, y_b=y_b, cat_idx=cat_idx)
            logits.append(class_logits)
        
        # Stack logits across all classes
        logits = torch.stack(logits, dim=1)

        # Separate foreground and background logits
        fg_logits = logits[:, :, 1].clone()
        bg_logits = logits[:, :, 0].clone()

        # Determine background positions
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))

        # Combine background and foreground logits
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        # Postprocess the logits
        logits = self.postprocess_masks(logits, batch["dims"])

        return {
            ResultDict.LOGITS: logits,
        }

        
        
def build_hdmnet(shots=1, val_fold_idx=0):
    args = EasyDict({
        "layers": 50,
        "vgg": False,
        "aux_weight1": 1.0,
        "aux_weight2": 1.0,
        "low_fea": 'layer2',  # low_fea for computing the Gram matrix
        "kshot_trans_dim": 2, # K-shot dimensionality reduction
        "merge": 'final',     # fusion scheme for GFSS ('base' Eq(S1) | 'final' Eq(18) )
        "merge_tau": 0.9,     # fusion threshold tau 
        "zoom_factor": 8,
        "shot": shots,
        "data_set": "coco",
        "ignore_label": 255,
        "print_freq": 10,
        "split": val_fold_idx,
    })
    model = HDMNetModel(args, cls_type="Base")

    # checkpoint_per_fold_1shot = {
    #     0: "checkpoints/bam/coco/split0/res50/train_epoch_43.5_0.4341.pth",
    #     1: "checkpoints/bam/coco/split1/res50/train_epoch_48_0.5059.pth",
    #     2: "checkpoints/bam/coco/split2/res50/train_epoch_44.5_0.4749.pth",
    #     3: "checkpoints/bam/coco/split3/res50/train_epoch_45_0.4342.pth",
    # }

    # checkpoint_per_fold_5shot = {
    #     0: "checkpoints/bam/coco/split0/res50/train5_epoch_47.5_0.4926.pth",
    #     1: "checkpoints/bam/coco/split1/res50/train5_epoch_45.5_0.5420.pth",
    #     2: "checkpoints/bam/coco/split2/res50/train5_epoch_45_0.5163.pth",
    #     3: "checkpoints/bam/coco/split3/res50/train5_epoch_47_0.4955.pth",
    # }
    # assert shots in [1, 5]
    # checkpoint_per_fold = checkpoint_per_fold_1shot if shots == 1 else checkpoint_per_fold_5shot
    # checkpoint_path = checkpoint_per_fold[val_fold_idx]    

    # if os.path.isfile(checkpoint_path):
    #     print("=> loading checkpoint '{}'".format(checkpoint_path))
    #     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    #     new_param = checkpoint['state_dict']
    #     try: 
    #         model.load_state_dict(new_param)
    #     except RuntimeError:                   # 1GPU loads mGPU model
    #         for key in list(new_param.keys()):
    #             new_param[key[7:]] = new_param.pop(key)
    #         model.load_state_dict(new_param)
    #     print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(checkpoint_path))
    return model
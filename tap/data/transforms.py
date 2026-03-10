from typing import Any
from torchvision.transforms.functional import resize
from PIL import Image
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
import numpy as np
from copy import deepcopy
from typing import Tuple
from tap.data.utils import get_preprocess_shape
from torchvision.transforms import Normalize as Norm
from torchvision.transforms import Resize
from skimage.segmentation import slic

class CustomResize(object):
    def __init__(self, long_side_length: int = 1024):
        self.long_side_length = long_side_length

    def __call__(self, sample: Image):
        """
        Resize the image to the target long side length.
        """
        oldw, oldh = sample.size if isinstance(sample, Image.Image) else sample.shape[-2:]
        target_size = get_preprocess_shape(oldh, oldw, self.long_side_length)
        return resize(sample, target_size)


class CustomNormalize(object):
    def __init__(
        self, long_side_length: int = 1024, mean: Any = [0.485, 0.456, 0.406], std: Any = [0.229, 0.224, 0.225]
    ):
        self.long_side_length = long_side_length
        self.pixel_mean = torch.tensor(mean).view(-1, 1, 1)
        self.pixel_std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample: torch.Tensor):
        """
        Normalize the image.
        """
        sample = (sample - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = sample.shape[-2:]
        padh = self.long_side_length - h
        padw = self.long_side_length - w
        sample = F.pad(sample, (0, padw, 0, padh))
        return sample
    
class Normalize(Norm):
    def __init__(self, mean: Any = [0.485, 0.456, 0.406], std: Any = [0.229, 0.224, 0.225], inplace=False):
        super().__init__(mean, std, inplace)
    
    
    
class Denormalize(object):
    def __init__(self, mean: Any = [0.485, 0.456, 0.406], std: Any = [0.229, 0.224, 0.225], device: Any = "cpu"):
        self.pixel_mean = torch.tensor(mean, device=device).view(-1, 1, 1)
        self.pixel_std = torch.tensor(std, device=device).view(-1, 1, 1)
        self.long_side_length = 1024

    def __call__(self, sample: torch.Tensor):
        """
        Denormalize the image.
        """
        sample = sample * self.pixel_std + self.pixel_mean
        return sample


class PromptsProcessor:
    def __init__(self, long_side_length: int = 1024, masks_side_length: int = 256, custom_preprocess=True):
        self.long_side_length = long_side_length
        self.masks_side_length = masks_side_length
        self.custom_preprocess=custom_preprocess

    def __ann_to_rle(self, ann, h, w):
        """Convert annotation which can be polygons, to RLE.

        Args:
            ann (dict): annotation object
            h (int): image height
            w (int): image width
        """
        segm = ann
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann
        return rle
    
    def __add_bbox_noise(self, bbox, hb, wb, h, w):
        x1, y1, x2, y2 = bbox
        # take random number from normal distribution with mean 0 and std 0.1 * l
        noise_x1 = np.clip(np.random.normal(0, 0.1 * wb), -20, 20)
        noise_y1 = np.clip(np.random.normal(0, 0.1 * hb), -20, 20)
        noise_x2 = np.clip(np.random.normal(0, 0.1 * wb), -20, 20)
        noise_y2 = np.clip(np.random.normal(0, 0.1 * hb), -20, 20)

        x1 = float(np.clip(x1 + noise_x1, 0, w))
        y1 = float(np.clip(y1 + noise_y1, 0, h))
        x2 = float(np.clip(x2 + noise_x2, 0, w))
        y2 = float(np.clip(y2 + noise_y2, 0, h))

        return [x1, y1, x2, y2]


    def convert_bbox(self, bbox, h, w, noise=False):
        # convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        x, y, wb, hb = bbox
        x1 = x
        y1 = y
        x2 = x + wb
        y2 = y + hb
        if noise:
            return self.__add_bbox_noise([x1, y1, x2, y2], hb, wb, h, w)
        return [x1, y1, x2, y2]

    def convert_mask(self, mask, h, w):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.
        Args:
            mask: mask can be polygons, uncompressed RLE, or RLE
            h (int): image height
            w (int): image width

        Returns:
            binary mask (numpy 2D array)
        """
        rle = self.__ann_to_rle(mask, h, w)
        matrix = mask_utils.decode(rle)
        # if matrix is made by all zeros
        if np.all(matrix == 0):
            if isinstance(mask, list):
                first_polygon = mask[0]
                fp_x, fp_y = int(first_polygon[0]), int(first_polygon[1])
                # check if fp_x and fp_y are within the image
                fp_x = min(fp_x, w - 1)
                fp_y = min(fp_y, h - 1)
                # check if fp_x and fp_y are negative
                fp_x = max(fp_x, 0)
                fp_y = max(fp_y, 0) 
                matrix[fp_y, fp_x] = 1
            else:
                matrix[0, 0] = 1
        return matrix

    def sample_point(self, mask: np.ndarray):
        # make a list of positive (row, col) coordinates
        positive_coords = np.argwhere(mask)
        # choose one at random
        row, col = positive_coords[np.random.choice(len(positive_coords))]
        return col, row

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        if self.custom_preprocess:
            new_h, new_w = get_preprocess_shape(original_size[0], original_size[1], self.long_side_length)
        else:
            new_h, new_w = self.long_side_length, self.long_side_length
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    def torch_apply_coords(
        self, coords: torch.tensor, original_size: Tuple[int, ...]
    ) -> torch.tensor:
        """
        Expects a torch of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        if self.custom_preprocess:
            new_h, new_w = get_preprocess_shape(original_size[0], original_size[1], self.long_side_length)
        else:
            new_h, new_w = self.long_side_length, self.long_side_length
        coords = coords.clone().float()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_masks(self, masks: np.ndarray):
        # take the binary OR over the masks and resize to new size
        if len(masks) == 0:
            return torch.zeros(
                (self.masks_side_length, self.masks_side_length), dtype=torch.uint8
            )

        mask = torch.as_tensor(np.logical_or.reduce(masks).astype(np.uint8)).unsqueeze(
            0
        )
        if self.custom_preprocess:
            new_h, new_w = get_preprocess_shape(masks[0].shape[0], masks[0].shape[1], self.long_side_length)
            mask = resize(mask, (new_h, new_w), interpolation=Image.NEAREST)
            padw = self.long_side_length - new_w
            padh = self.long_side_length - new_h
            mask = F.pad(mask, (0, padw, 0, padh))
        mask = resize(
            mask,
            (self.masks_side_length, self.masks_side_length),
            interpolation=Image.NEAREST,
        )
        return mask


class SuperpixelMaskPerturbator:
    def __init__(self, perturbation_ratio: float = 0.1, n_segments: int = 100, compactness: float = 10.0):
        """
        Args:
            perturbation_ratio (float): Percentage of foreground superpixels to remove per class.
            n_segments (int): Approximate number of superpixels to divide the image into.
            compactness (float): Balances color proximity versus space for SLIC.
        """
        self.perturbation_ratio = perturbation_ratio
        self.n_segments = n_segments
        self.compactness = compactness

    def perturb(self, batch: dict) -> dict:
        # Clone the mask to avoid modifying the original tensor in place
        perturbed_mask = batch["prompt_masks"].clone()
        ground_truth_mask = batch["ground_truths"]
        images = batch["images"]  
        
        M, C_mask, H, W = perturbed_mask.shape

        for m in range(M):
            # Extract the specific image [3, H, W]
            img_tensor = images[m]

            # 1. Prepare image for SLIC (from Tensor [3, H, W] to Numpy [H, W, 3])
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)

            # 2. Generate superpixels (computed once per image, reused for all classes)
            segments = slic(img_np, n_segments=self.n_segments, compactness=self.compactness, start_label=1)
            segments_tensor = torch.from_numpy(segments).to(perturbed_mask.device)

            # 3. Iterate over each class channel independently
            for c in range(C_mask):
                current_mask = perturbed_mask[m, c] 
                foreground_mask = current_mask == 1
                
                # Identify which superpixels overlap with the foreground of this specific class
                foreground_segments = segments_tensor[foreground_mask]
                
                if foreground_segments.numel() == 0:
                    continue # Skip if there is no foreground for this class

                unique_fg_segments = torch.unique(foreground_segments)

                # 4. Calculate the number of regions to remove for this class
                num_regions_to_perturb = int(len(unique_fg_segments) * self.perturbation_ratio)

                if num_regions_to_perturb > 0:
                    # 5. Randomly select and remove regions
                    perm = torch.randperm(len(unique_fg_segments))
                    selected_regions = unique_fg_segments[perm[:num_regions_to_perturb]]

                    # 6. Apply perturbation (flip 1 to 0) for the selected superpixels
                    mask_to_remove = torch.isin(segments_tensor, selected_regions)
                    perturbed_mask[m, c][mask_to_remove] = 0
                    if m > 0: # Only apply perturbation to support set ground truth masks, not query set
                        ground_truth_mask[m][mask_to_remove] = 0

        # Update the batch dictionary
        batch["prompt_masks"] = perturbed_mask
        batch["ground_truths"] = ground_truth_mask
        return batch
    
    def __call__(self, batch: dict) -> dict:
        return self.perturb(batch)
import numpy as np
import albumentations as A
import torch

from einops import rearrange

from tap.data.transforms import PromptsProcessor
from tap.data.utils import BatchKeys


def divide_query_examples(
    batch,
    ground_truths,
    torch_keys_to_exchange,
    list_keys_to_exchange,
    torch_keys_to_separate,
    list_keys_to_separate,
    torch_keys_to_not_subsample,
    subsample=None,
):
    batch_examples = {key: batch[key][:, 1:] for key in torch_keys_to_separate}
    for key in list_keys_to_separate:
        batch_examples[key] = [elem[1:] for elem in batch[key]]
    gt = ground_truths[:, 0]
    for key in batch.keys() - set(torch_keys_to_separate + list_keys_to_separate):
        batch_examples[key] = batch[key]

    if subsample:
        support_set_len = batch[BatchKeys.IMAGES].shape[1] - 1
        device = batch["images"].device
        granted_first_sample = torch.tensor([0], device=device)
        index_tensor = torch.randperm(support_set_len - 1, device=device)[
            : subsample - 1
        ]
        index_tensor = torch.cat([granted_first_sample, index_tensor])
        query_index_tensor = torch.cat(
            [torch.tensor([0], device=device), index_tensor + 1]
        )

        for key_set, separate_keys in [
            (torch_keys_to_exchange, torch_keys_to_separate),
            (list_keys_to_exchange, list_keys_to_separate),
        ]:
            for key in key_set:
                if key in batch_examples and key not in torch_keys_to_not_subsample:
                    indices = (
                        index_tensor if key in separate_keys else query_index_tensor
                    )
                    if isinstance(batch_examples[key], list):
                        batch_examples[key] = [
                            [elem[idx] for idx in indices] for elem in batch_examples[key]
                        ]
                    else:
                        batch_examples[key] = batch_examples[key][:, indices]

    return batch_examples, gt


def generate_incremental_tensors(N, K):
    example_matrix = torch.arange(N * K).reshape(N, K).T
    ordered_elements = example_matrix.flatten().tolist()
    result_tensors = []

    for i, num in enumerate(ordered_elements):
        # Start the tensor with the current element i (starting from 0)
        current_tensor = torch.tensor([num], dtype=torch.int32)

        # Determine how many extra elements to include based on iteration
        # Each tensor has 1 + (extra elements). The number of extra elements grows by N every N iterations
        num_extra_rows = (
            i // N
        ) + 1  # Extra elements = N for first batch, 2N for second, etc.

        i_row = i // N  # The row of the current element in the example matrix
        if i_row == K - 1:
            # If the current element is the last element in the example matrix, skip
            # because there are no more rows to sample from
            extra_elements = torch.tensor([x for x in ordered_elements if x != num])
        else:
            possible_rows = [row_k for row_k in range(K) if row_k != i_row]
            sampled_rows = np.random.choice(
                possible_rows, num_extra_rows, replace=False
            )
            sampled_rows = torch.tensor(np.sort(sampled_rows))
            rows = torch.index_select(example_matrix, 0, sampled_rows)
            extra_elements = rearrange(rows, "b n -> (b n)", n=N)

        # Combine the starting element with the extra elements
        current_tensor = torch.cat((current_tensor, extra_elements))

        # Add the tensor to the result list
        result_tensors.append(current_tensor)

    return result_tensors

def augment_support_set(batch, ground_truths, k_augmented=5):
    """
    Augment the support set until the number of support images is k_augmented times.
    """
    current_support_set_len = batch[BatchKeys.IMAGES].shape[1]
    if current_support_set_len >= k_augmented:
        return batch
    augment_times = k_augmented - current_support_set_len
    
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.8),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.8),
        A.RandomBrightnessContrast(p=0.8),
    ])
    
    for i in range(augment_times):
        imgs = batch[BatchKeys.IMAGES][:, i % current_support_set_len].cpu().numpy()
        masks = batch[BatchKeys.PROMPT_MASKS][:, i % current_support_set_len]
        gts = ground_truths[:, i % current_support_set_len]
        
        B = imgs.shape[0]
        
        imgs_augmented = []
        masks_augmented = []
        gt_augmented = []
        
        for b in range(B):
            img = imgs[b]
            mask = masks[b]
            ground_truth = gts[b]
            mask_hw = mask.shape[1:]
            gt_hw = ground_truth.shape[-2:]
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=img.shape[1:], mode='nearest').squeeze().cpu().numpy()
            ground_truth = torch.nn.functional.interpolate(ground_truth.unsqueeze(0).unsqueeze(0).float(), size=img.shape[1:], mode='nearest').squeeze().cpu().numpy()
            
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC
            mask = np.transpose(mask, (1, 2, 0))  # Convert to HWC
            # ground_truth = np.transpose(ground_truth, (1, 2, 0)) h # Convert to HWC
            augmented = augmentations(image=img, mask=mask, mask0=ground_truth)
            aug_img = augmented['image']
            aug_img_tensor = torch.tensor(np.transpose(aug_img, (2, 0, 1))).unsqueeze(0).to(batch[BatchKeys.IMAGES].device)
            # Mask
            aug_mask = augmented['mask']
            aug_mask = np.transpose(aug_mask, (2, 0, 1))  # Convert back to CHW
            aug_mask = torch.nn.functional.interpolate(torch.tensor(aug_mask).unsqueeze(0), size=mask_hw, mode='nearest')
            aug_mask = aug_mask.to(batch[BatchKeys.PROMPT_MASKS].device)
            # GT
            aug_gt = augmented['mask0']
            # aug_gt = np.transpose(aug_gt, (2, 0, 1))  # Convert back to CHW
            aug_gt = torch.nn.functional.interpolate(torch.tensor(aug_gt).unsqueeze(0).unsqueeze(0), size=gt_hw, mode='nearest')
            aug_gt = aug_gt.to(ground_truths.device)
            gt_augmented.append(aug_gt)
            
            imgs_augmented.append(aug_img_tensor)
            masks_augmented.append(aug_mask)

        imgs_augmented = torch.cat(imgs_augmented, dim=0)
        masks_augmented = torch.cat(masks_augmented, dim=0)
        gt_augmented = torch.cat(gt_augmented, dim=0).long()
        
        ground_truths = torch.cat([ground_truths, gt_augmented], dim=1)

        batch[BatchKeys.IMAGES] = torch.cat([batch[BatchKeys.IMAGES], imgs_augmented.unsqueeze(1)], dim=1)
        batch[BatchKeys.PROMPT_MASKS] = torch.cat([batch[BatchKeys.PROMPT_MASKS], masks_augmented.unsqueeze(1)], dim=1)

        for key in [BatchKeys.FLAG_MASKS, BatchKeys.FLAG_EXAMPLES, BatchKeys.FLAG_GTS]:
            if key in batch:
                mask = batch[key][:, i % current_support_set_len].unsqueeze(1)
                batch[key] = torch.cat([batch[key], mask], dim=1)
        
        if BatchKeys.DIMS in batch:
            dims = batch[BatchKeys.DIMS][:, i % current_support_set_len].unsqueeze(1)
            batch[BatchKeys.DIMS] = torch.cat([batch[BatchKeys.DIMS], dims], dim=1)
        
        for key in [BatchKeys.CLASSES, BatchKeys.IMAGE_IDS]:
            if key in batch:
                elem = [batch[key][k][i % current_support_set_len] for k in range(B)]
                batch[key] = [batch[key][k] + [elem[k]] for k in range(B)]
                
    return batch, ground_truths


class Substitutor:
    """
    A class that cycle all the images in the examples as a query image.
    """

    torch_keys_to_exchange = [
        BatchKeys.IMAGES,
        BatchKeys.PROMPT_MASKS,
        BatchKeys.FLAG_MASKS,
        BatchKeys.FLAG_EXAMPLES,
        BatchKeys.DIMS,
    ]
    torch_keys_to_separate = [
        BatchKeys.PROMPT_MASKS,
        BatchKeys.FLAG_MASKS,
        BatchKeys.FLAG_EXAMPLES,
    ]
    list_keys_to_exchange = [BatchKeys.CLASSES, BatchKeys.IMAGE_IDS]
    list_keys_to_separate = []
    torch_keys_to_not_subsample = [BatchKeys.DIMS]

    def __init__(
        self,
        substitute=True,
        long_side_length=1024,
        custom_preprocess=True,
        subsample=None,
        augment=False,
        k_augmented=5,
        **kwargs,
    ) -> None:
        if kwargs:
            print(f"Warning: Unrecognized arguments: {kwargs}")
        self.example_classes = None
        self.substitute = substitute
        self.it = 0
        self.query_iteration = True
        self.subsample = subsample
        self.prompt_processor = PromptsProcessor(
            long_side_length=long_side_length, custom_preprocess=custom_preprocess
        )
        self.augment = augment
        self.k_augmented = k_augmented

    def reset(self, batch: dict) -> None:
        self.it = 0
        self.query_iteration = True
        self.batch, self.ground_truths = batch
        self.batch, self.query_image_gt_dim = self.first_divide_query_examples()
        self.example_classes = self.batch[BatchKeys.CLASSES]
        if self.augment:
            raise NotImplementedError("Augmentation is not implemented yet.")
            self.batch, self.ground_truths = augment_support_set(self.batch, self.ground_truths, self.k_augmented)

    def __iter__(self):
        return self

    def first_divide_query_examples(self):
        batch_examples, query_gt = self.divide_query_examples()
        self.ground_truths = self.ground_truths[:, 1:]
        query_image = batch_examples["images"][:, 0]
        batch_examples["images"] = batch_examples["images"][:, 1:]
        query_dims = batch_examples["dims"][:, 0]
        batch_examples["dims"] = batch_examples["dims"][:, 1:]
        return batch_examples, (query_image, query_gt, query_dims)

    def divide_query_examples(self, batch=None, gt=None):
        batch = self.batch if batch is None else batch
        gt = self.ground_truths if gt is None else gt
        return divide_query_examples(
            batch,
            gt,
            self.torch_keys_to_exchange,
            self.list_keys_to_exchange,
            self.torch_keys_to_separate,
            self.list_keys_to_separate,
            self.torch_keys_to_not_subsample,
            subsample=self.subsample if not self.query_iteration else None, # Only subsample after first iteration
        )

    def divide_query_examples_append_query_dim(self):
        batch, gt = self.divide_query_examples()
        batch = {
            **batch,
            "dims": torch.cat(
                [batch["dims"], self.query_image_gt_dim[2].unsqueeze(1)], dim=1
            ),
        }
        return batch, gt
    
    def same_query_support_iteration(self):
        """
        In case of single image in the batch, the only support image will duplicated also as query.
        """
        batch = {
            **self.batch,
            "images": torch.cat([self.batch["images"], self.batch["images"]], dim=1),
            "dims": torch.cat([self.batch["dims"], self.batch["dims"], self.query_image_gt_dim[2].unsqueeze(1)], dim=1),
        }
        return batch, self.ground_truths[0]

    def get_batch_info(self):
        num_images = self.batch["images"].shape[1]
        device = self.batch["images"].device
        return num_images, device

    def _query_iteration(self):
        self.query_iteration = False
        batch = {
            **self.batch,
            "images": torch.cat(
                [self.query_image_gt_dim[0].unsqueeze(1), self.batch["images"]], dim=1
            ),
            "dims": torch.cat(
                [self.query_image_gt_dim[2].unsqueeze(1), self.batch["dims"]], dim=1
            ),
        }
        return batch, self.query_image_gt_dim[1]

    def __next__(self):
        num_images, device = self.get_batch_info()

        if self.query_iteration:
            return self._query_iteration()
        if self.it == 0:
            self.it = 1
            if num_images == 1:
                return self.same_query_support_iteration()
            else:
                return self.divide_query_examples_append_query_dim()

        if not self.substitute:
            raise StopIteration
        if self.it == num_images:
            raise StopIteration
        else:
            index_tensor = torch.cat(
                [
                    torch.tensor([self.it], device=device),
                    torch.arange(0, self.it, device=device),
                    torch.arange(self.it + 1, num_images, device=device),
                ]
            ).long()

        for key in self.torch_keys_to_exchange:
            self.batch[key] = torch.index_select(
                self.batch[key], dim=1, index=index_tensor
            )

        for key in self.list_keys_to_exchange:
            self.batch[key] = [
                [elem[i] for i in index_tensor] for elem in self.batch[key]
            ]
        for key in self.batch.keys() - set(
            self.torch_keys_to_exchange + self.list_keys_to_exchange
        ):
            self.batch[key] = self.batch[key]

        self.ground_truths = torch.index_select(
            self.ground_truths, dim=1, index=index_tensor
        )

        self.it += 1
        return self.divide_query_examples_append_query_dim()


class IncrementalSubstitutor(Substitutor):
    def __init__(
        self,
        substitute=True,
        long_side_length=1024,
        custom_preprocess=True,
        n_ways=None,
        k_shots=None,
    ):
        super().__init__(substitute, long_side_length, custom_preprocess)
        self.n_ways = n_ways
        self.k_shots = k_shots
        super().__init__(substitute, long_side_length, custom_preprocess)
        self.index_tensors = generate_incremental_tensors(self.n_ways, self.k_shots)

    def reset(self, batch):
        self.index_tensors = generate_incremental_tensors(self.n_ways, self.k_shots)
        return super().reset(batch)

    def divide_query_examples_append_dims(self, batch, gt):
        batch = {
            **batch,
            "dims": torch.cat(
                [batch["dims"], self.query_image_gt_dim[2].unsqueeze(1)], dim=1
            ),
        }
        return self.divide_query_examples(batch, gt)

    def __next__(self):
        num_images, device = self.get_batch_info()
        torch_keys_to_exchange = self.torch_keys_to_exchange.copy()
        torch_keys_to_exchange.remove(BatchKeys.DIMS)
        if self.query_iteration:
            return self._query_iteration()
        if not self.substitute:
            raise StopIteration
        if self.it == num_images:
            raise StopIteration
        else:
            index_tensor = self.index_tensors[self.it].to(device)

        new_batch = {
            key: torch.index_select(self.batch[key], dim=1, index=index_tensor)
            for key in torch_keys_to_exchange
        }
        for key in self.list_keys_to_exchange:
            new_batch[key] = [
                [elem[i] for i in index_tensor] for elem in self.batch[key]
            ]
        # Add the remaning DIMS
        dims_index = torch.cat(
            (
                index_tensor,
                torch.tensor(
                    [x for x in range(num_images) if x not in index_tensor],
                    device=device,
                    dtype=torch.long,
                ),
            )
        )
        new_batch[BatchKeys.DIMS] = torch.index_select(
            self.batch[BatchKeys.DIMS], dim=1, index=dims_index
        )

        for key in self.batch.keys() - set(
            torch_keys_to_exchange + self.list_keys_to_exchange
        ):
            new_batch[key] = self.batch[key]

        new_gt = torch.index_select(self.ground_truths, dim=1, index=index_tensor)

        self.it += 1
        return self.divide_query_examples_append_dims(new_batch, new_gt)

import copy
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tap.data.transforms import Normalize, Resize

from tap.data.dataset import FSSDataset, VariableBatchSampler
from tap.data.coco import CocoLVISTestDataset, CocoLVISDataset
from tap.data.transforms import CustomNormalize, CustomResize
from tap.data.utils import get_mean_std


def map_collate(dataset):
    return dataset.collate_fn if hasattr(dataset, "collate_fn") else None


def get_preprocessing(params):
    SIZE = 256
    size = params.get("common", {}).get("image_size", SIZE)
    if "preprocess" in params.get("common", {}):
        preprocess_params = params["common"].pop("preprocess")
        mean = preprocess_params["mean"]
        std = preprocess_params["std"]
        mean, std = get_mean_std(mean, std)
    else:
        mean, std = get_mean_std("default", "default")
    preprocess = Compose(
            [
                Resize(size=(size, size)),
                ToTensor(),
                Normalize(mean, std),
            ]
        )
    return preprocess


def get_dataloaders(dataset_args, dataloader_args, num_processes):
    preprocess = get_preprocessing(dataset_args)
    dataloader_args = copy.deepcopy(dataloader_args)

    datasets_params = dataset_args.get("datasets")
    common_params = dataset_args.get("common")
    possible_batch_example_nums = dataloader_args.pop("possible_batch_example_nums")
    val_possible_batch_example_nums = dataloader_args.pop(
        "val_possible_batch_example_nums", possible_batch_example_nums
    )

    prompt_types = dataloader_args.pop("prompt_types", None)
    prompt_choice_level = dataloader_args.pop("prompt_choice_level", "batch")

    val_prompt_types = dataloader_args.pop("val_prompt_types", prompt_types)

    val_datasets_params = {
        k: v for k, v in datasets_params.items() if k.startswith("val_")
    }

    val_dataloaders = {}
    for dataset, params in val_datasets_params.items():
        splits = dataset.split("_")
        if len(splits) > 2:
            dataset_name = "_".join(splits[:2])
        else:
            dataset_name = dataset
        val_dataset = FSSDataset(
            datasets_params={dataset_name: params},
            common_params={**common_params, "preprocess": preprocess},
        )
        val_batch_sampler = VariableBatchSampler(
            val_dataset,
            possible_batch_example_nums=val_possible_batch_example_nums,
            num_processes=num_processes,
            prompt_types=val_prompt_types,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            **dataloader_args,
            collate_fn=val_dataset.collate_fn,
            batch_sampler=val_batch_sampler,
        )
        val_dataloaders[dataset] = val_dataloader

    return val_dataloaders

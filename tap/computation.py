import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import sys
import pandas as pd
import wandb

from tap.adapters import get_peft_config, get_peft_model
from tap.validate import DATASETS, get_model, dataloader_args, dataset_args, get_dataloaders, substitutor_cls
from tap.data.utils import BatchKeys

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()

def get_state_size(state_dict):
    if isinstance(state_dict, dict):
        return sum(get_state_size(v) for v in state_dict.values())
    elif isinstance(state_dict, list):
        return sum(get_state_size(v) for v in state_dict)
    elif isinstance(state_dict, torch.Tensor):
        return state_dict.numel() * state_dict.element_size()
    else:
        return 0


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_name: str, params: dict, peft_type, lora_params: dict):
    lora_config = get_peft_config(peft_type, lora_params)
    model, image_size = get_model(model_name, **params)
    model = get_peft_model(model, lora_config)
    return model, image_size


def evaluation_step(model, batch, gt, optimizer, device):
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device=device)

    # Memory before forward
    mem_before_forward = torch.cuda.max_memory_allocated(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)
    sync_cuda()
    start_forward = torch.cuda.Event(enable_timing=True)
    end_forward = torch.cuda.Event(enable_timing=True)
    start_forward.record()

    res = model(batch)["logits"]

    end_forward.record()
    sync_cuda()
    mem_after_forward = torch.cuda.max_memory_allocated(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)

    # Backward pass
    sync_cuda()
    start_backward = torch.cuda.Event(enable_timing=True)
    end_backward = torch.cuda.Event(enable_timing=True)
    start_backward.record()

    loss = F.cross_entropy(res, gt)
    loss.backward()

    end_backward.record()
    sync_cuda()
    mem_after_backward = torch.cuda.max_memory_allocated(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)

    # Optimizer step
    sync_cuda()
    start_step = torch.cuda.Event(enable_timing=True)
    end_step = torch.cuda.Event(enable_timing=True)
    start_step.record()

    optimizer.step()
    optimizer.zero_grad()

    end_step.record()
    sync_cuda()
    mem_after_step = torch.cuda.max_memory_allocated(device=device)

    opt_state_size = get_state_size(optimizer.state_dict())

    # Calculate elapsed times in milliseconds
    time_forward = start_forward.elapsed_time(end_forward) / 1000
    time_backward = start_backward.elapsed_time(end_backward) / 1000
    time_step = start_step.elapsed_time(end_step) / 1000

    return {
        "mem_before_forward": mem_before_forward,
        "mem_after_forward": mem_after_forward,
        "mem_after_backward": mem_after_backward,
        "mem_after_step": mem_after_step,
        "opt_state_size": opt_state_size,
        "time_forward": time_forward,
        "time_backward": time_backward,
        "time_step": time_step,
    }


def run_test(params: str, use_lora: bool):
    
    mode = 'with LoRA' if use_lora else 'without LoRA'
    print(f"Running test {mode}...")

    n_steps = params.get("n_steps", 100)
    warmup_steps = params.get("warmup_steps", 10)

    model_name = params["model"]
    
    batch_size = 1
    peft_type = params["peft_type"] if use_lora else "full"
    n_ways = params["n_ways"]
    k_shots = params["k_shots"]
    val_num_samples = (n_steps+warmup_steps) * batch_size
    val_fold_idx = params["val_fold_idx"]
    dataset = params["dataset"]
    device = torch.device(params.get("device", "cuda"))
    substitutor = "default"
    
    model_params = {
        "dataset": dataset,
        "val_fold_idx": val_fold_idx,
        "k_shots": k_shots,
        "n_ways": n_ways,
    }
    lora_params = {
        "lora_r": params.get("lora_r", 64),
        "lora_alpha": params.get("lora_alpha", None),
        "target_modules": params.get("target_modules", None),
        "lora_dropout": params.get("lora_dropout", 0.05),
    }

    model, image_size = build_model(model_name, model_params, peft_type, lora_params)
    model = model.to(device)

    print(f"Target modules: {model.targeted_module_names}")

    substitutor = substitutor_cls[substitutor](
        substitute=True,
        long_side_length=480,
        custom_preprocess=False,
        n_ways=n_ways,
        k_shots=k_shots,
        subsample=None,
    )
    

    dataset_name, datasets_params = DATASETS[dataset]
    dataset_args["datasets"][dataset_name] = datasets_params

    dataset_args["datasets"][dataset_name]["n_ways"] = n_ways
    dataset_args["datasets"][dataset_name]["n_shots"] = k_shots
    dataset_args["datasets"][dataset_name]["val_num_samples"] = val_num_samples
    dataset_args["datasets"][dataset_name]["val_fold_idx"] = val_fold_idx
    dataset_args["common"]["image_size"] = image_size

    val_dict = get_dataloaders(
        dataset_args, dataloader_args, num_processes=1
    )
    val = val_dict[dataset_name]
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    results = []
    
    for i, (batch_tuple, data_name) in zip(range(warmup_steps+n_steps), val):
        print(f"Step {i+1}/{warmup_steps+n_steps}", end='\r')
    
        batch_gt = batch_tuple[1].to(device)
        batch_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_tuple[0].items()
        }
        batch_tuple = (batch_dict, batch_gt)
        substitutor.reset(batch=batch_tuple)
        
        (batch, gt) = next(iter(substitutor))
        step_result = evaluation_step(model, batch, gt, optimizer, device)
        
        if i >= warmup_steps:
            results.append(step_result)
            
    # Aggregate results
    df = pd.DataFrame(results)
    # Aggregate over rows
    agg_results = df.mean().to_dict()
    
    trainable_params = count_trainable_params(model)

    # Print structured table
    print(f"{'Stage':<20} {'Memory (MB)':<15} {'Time (s)':<10}")
    print("="*50)
    print(f"{'Before Forward':<20} {agg_results['mem_before_forward']/1e6:<15.2f} {'-':<10}")
    print(f"{'After Forward':<20} {agg_results['mem_after_forward']/1e6:<15.2f} {agg_results['time_forward']:<10.4f}")
    print(f"{'After Backward':<20} {agg_results['mem_after_backward']/1e6:<15.2f} {agg_results['time_backward']:<10.4f}")
    print(f"{'After Step':<20} {agg_results['mem_after_step']/1e6:<15.2f} {agg_results['time_step']:<10.4f}")
    print("="*50)
    print(f"Trainable Params: {trainable_params}")
    print(f"Optimizer State Size: {agg_results['opt_state_size']/1e6:.2f} MB")
    
    return {**agg_results, "trainable_params": trainable_params}


def computation_test(params):

    group = params.get("experiment", {}).get("group", "Computation")

    if params.get("lora", True):
        tracker = wandb.init(project="lorafss", config={**params, "lora": True}, group=group)
        with_lora_test = run_test(params, use_lora=True)
        tracker.log(with_lora_test)
        tracker.finish()
    else:
        print("Skipping test without LoRA since lora is set to False in parameters.")
        
    if params.get("lora", True):
        params["target_modules"] = None  # Use full model for the test without LoRA

    tracker = wandb.init(project="lorafss", config={**params, "lora": False}, group=group)
    without_lora_test = run_test(params, use_lora=False)
    tracker.log(without_lora_test)
    tracker.finish()

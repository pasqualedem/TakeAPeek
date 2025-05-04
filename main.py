import copy
import os
import subprocess
import sys
import uuid
import click

from tap.utils.grid import linearize, make_grid
from tap.utils.utils import get_timestamp, load_yaml, nested_dict_update, write_yaml
from tap.validate import main as lora_validate_main
from tap.logger.text_logger import get_logger

logger = get_logger(__name__)


def experiment_lora(param_file, multi_gpu=False, only_create=False, sequential=False):
    timestamp = get_timestamp()
    settings = load_yaml(param_file)
    base_grid = settings["parameters"]
    other_grids = settings["other_grids"]
    
    if "experiment" not in settings:
        settings["experiment"] = {}
    if "group" not in settings["experiment"]:
        settings["experiment"]["group"] = "LoRa"

    print("\n" + "=" * 100)
    complete_grids = [base_grid]
    if other_grids:
        complete_grids += [
            nested_dict_update(copy.deepcopy(base_grid), other_run)
            for other_run in other_grids
        ]

    grids, dot_elements = zip(
        *[
            make_grid(grid, return_cartesian_elements=True)
            for grid in complete_grids
        ]
    )
    # WARNING: Grids' objects have the same IDs!
    dot_elements = list(dot_elements)
    if len(dot_elements) > 1:
        dot_elements[1:] = [
            list(dict(linearize(others) + dot).items())
            for others, dot in zip(other_grids, dot_elements[1:])
        ]

    for i, grid in enumerate(grids):
        print(f"Grid {i+1}:")
        for j, run_params in enumerate(grid):
            print(f"Run {j+1}:")
            if sequential:
                run_params['experiment'] = settings["experiment"]
                if "," in run_params.get("target_modules", []):
                    run_params["target_modules"] = run_params["target_modules"].split(",")
                    run_params["target_modules"] = [mod for mod in run_params["target_modules"] if len(mod) > 0]
                main(copy.deepcopy(run_params))
            else:
                raise NotImplementedError("Parallel runs are not implemented yet.")


@click.command()
@click.option("--num_iterations", default=10, help="Number of iterations to run.")
@click.option("--device", default="cuda", help="Device to use (e.g., cuda or cpu).")
@click.option("--lora_r", default=32, help="LoRA r parameter.", type=int)
@click.option("--lora_alpha", default=None, help="LoRA alpha parameter. If None will be set to r", type=float)
@click.option("--lr", default=1e-4, help="Learning rate.", type=float)
@click.option("--dataset", default="coco", help="Dataset to use (coco or pascal).")
@click.option("--model", default="tap", help="Model to use.")
@click.option(
    "--target_modules",
    default="query,value",
    help="Comma-separated list of target modules.",
)
@click.option(
    "--substitutor",
    default="default",
    help="Substitutor to use. (default or incremental)",
)
@click.option(
    "--n_ways", default=2, help="Number of classes of the FSS validation", type=int
)
@click.option(
    "--k_shots",
    default=5,
    help="Number of shots per class of the FSS validation",
    type=int,
)
@click.option("--val_num_samples", default=100, help="Number of samples for validation.")
@click.option("--val_fold_idx", default=3, help="Fold index for validation.", type=int)
@click.option("--peft_type", default="lora", help="PEFT type to use (lora, xlora, loha, lokr, adalora)")
@click.option("--lora_dropout", default=0.1, help="LoRA dropout value.", type=float)
@click.option("--subsample", default=None, help="Subsample substitutor value.", type=int)
@click.option(
    "--experiment_file",
    default=None,
    help="Path to the file containing the parameters for the experiment, launching multiple parallel runs",
)
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a single run",
)
@click.option("--multi_gpu", is_flag=True, show_default=True, default=False, help="Use multiple GPUs")
@click.option("--only_create", is_flag=True, show_default=True, default=False, help="Only create the SLURM scripts")
@click.option("--sequential", is_flag=True, show_default=True, default=False, help="Run the experiments sequentially")
def cli(
    num_iterations,
    device,
    lora_r,
    lora_alpha,
    lr,
    dataset,
    model,
    peft_type,
    target_modules,
    lora_dropout,
    subsample,
    substitutor,
    n_ways,
    k_shots,
    val_num_samples,
    val_fold_idx,
    experiment_file,
    parameters,
    multi_gpu,
    only_create,
    sequential,
):
    """
    Command-line interface for setting parameters for LoRA training or testing.
    Collects parameters and passes them as a dictionary.
    """
    if experiment_file is not None:
        experiment_lora(experiment_file, multi_gpu, only_create=only_create, sequential=sequential)
        return

    if parameters is not None:
        params = load_yaml(parameters)
        if "," in params.get("target_modules", []):
            params["target_modules"] = params["target_modules"].split(",")
            params["target_modules"] = [mod for mod in params["target_modules"] if len(mod) > 0]
    else:
        # Convert the target_modules string to a list
        if "," in target_modules:
            target_modules = target_modules.split(",")
            target_modules = [mod for mod in target_modules if len(mod) > 0]

        # Create the parameters dictionary
        params = {
            "num_iterations": num_iterations,
            "device": device,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lr": lr,
            "target_modules": target_modules,
            "lora_dropout": lora_dropout,
            "subsample": subsample,
            "substitutor": substitutor,
            "n_ways": n_ways,
            "k_shots": k_shots,
            "val_num_samples": val_num_samples,
            "val_fold_idx": val_fold_idx,
            "model": model,
            "dataset": dataset,
            "peft_type": peft_type,
        }

    # Call the main function with the dictionary of parameters
    main(params)


def main(params):
    """
    Main function that accepts a dictionary of parameters.
    """
    # Print the parameters for debugging
    click.echo("Running with the following parameters:")
    for key, value in params.items():
        click.echo(f"{key}: {value}")

    lora_validate_main(params)


if __name__ == "__main__":
    cli()

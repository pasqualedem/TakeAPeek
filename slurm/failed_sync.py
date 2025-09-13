import os
import re
import click
import yaml
import wandb


PREFIX = "wandb: "

def failed_sync_out(path, out_file):
    print(f'################ Syncing file {os.path.join(path, out_file)} ################')
    with open(os.path.join(path, out_file), "r", errors="ignore") as f:
        lines = f.readlines()

    pattern = re.compile(
        r"miou0=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
        r"miouN=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)]"
    )

    miou0 = None
    miouN = None
    for line in reversed(lines):
        if "miou0" in line and "miouN" in line:

            match = pattern.search(line)
            if match:
                miou0 = float(match.group(1))
                miouN = float(match.group(2))
                break

    print(f"line: {line.strip()}")
            
    if miou0 is None or miouN is None:
        print("Error parsing miou0 or miouN")
        return False

    with open(os.path.join(path, out_file.replace('.out', '.yaml')), "r") as yaml_file:
        params = yaml.safe_load(yaml_file.read())
    if "," in params.get("target_modules", []):
        params["target_modules"] = params["target_modules"].split(",")
        params["target_modules"] = [mod for mod in params["target_modules"] if len(mod) > 0]
        
    if "num_iterations" not in params:
        print("num_iterations not in params")
        return False
    N = params["num_iterations"] - 1
        
    group = params.get("experiment", {}).get("group", None)
    tracker = wandb.init(project="lorafss", config=params, group=group, tags=["Restored"])
    
    tracker.log({"miou_orig": miou0})
    tracker.log({f"miou_it_{N}": miouN})
    tracker.log({f"gain_it_{N}": miouN - miou0})
    tracker.log({"miou": miouN})
    tracker.log({"gain": miouN - miou0})
    tracker.log({"saved_line": line.strip()})
    
    tracker.finish()
    
    return True

@click.command()
@click.argument('path', type=click.Path(exists=True))
def failed_sync(path):
    if os.path.isfile(path):
        if path.endswith('.out'):
            failed_sync_out(os.path.dirname(path), os.path.basename(path))
        else:
            print("Provided file is not an .out file")
    elif os.path.isdir(path):
        failed_sync_folder(path)
    else:
        print("Provided path is neither a file nor a directory")


def failed_sync_folder(path):
    print(f'Syncing folder {path}')
    files = os.listdir(path)
    out_files = [f for f in files if f.endswith('.out')]
    outcomes = []
    for out_file in out_files:
        outcome = failed_sync_out(path, out_file)
        outcomes.append(outcome)
    print(f"Successfully synced {sum(outcomes)} out of {len(outcomes)} files.")


if __name__ == "__main__":
    failed_sync()
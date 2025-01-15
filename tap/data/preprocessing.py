import os.path

import pandas as pd
import json


def transform_bbox_to_coords(bbox):
    x, y, w, h = bbox
    return [x, y, x+w, y+h]


def rename_coco20i_json(instances_path: str):
    """Change image filenames of COCO 2014 instances.

    Args:
        instances_path (str): Path to the COCO 2014 instances file.
    """
    with open(instances_path, "r") as f:
        anns = json.load(f)
    for image in anns["images"]:
        image["file_name"] = image["file_name"].split("_")[-1]
    with open(instances_path, "w") as f:
        json.dump(anns, f)
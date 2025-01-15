import click

@click.command()
@click.option(
    "--instances_path",
    default="data/annotations/instances_train2014.json",
    help="Path to the instances file",
)
def rename_coco20i_json_cli(instances_path):
    from tap.data.preprocessing import rename_coco20i_json
    print(f"Renaming COCO 2014 instances... {instances_path}")
    rename_coco20i_json(instances_path)
    
    
if __name__ == "__main__":
    rename_coco20i_json_cli()
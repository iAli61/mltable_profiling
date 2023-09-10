import yaml
import argparse
import os

def dump_mltable(output_folder):
    """
    Dump MLTable to a YAML file.

    Args:
        output_folder (str): The path to the output folder.

    Returns:
        None
    """
    # Define the paths dictionary
    d = {"paths": [{"file": "./mnist"}]}

    # Define the command line arguments parser
    parser = argparse.ArgumentParser(allow_abbrev=False, description="dump mltable")

    # Add the output_folder argument to the parser
    parser.add_argument("--output_folder", type=str, default=0)

    # Parse the command line arguments
    args, _ = parser.parse_known_args()

    # Define the path to the dump file
    dump_path = os.path.join(args.output_folder, "MLTable")

    # Dump the paths dictionary to the YAML file
    with open(dump_path, "w") as yaml_file:
        yaml.dump(d, yaml_file, default_flow_style=False)

    # Print a message to indicate that the MLTable file has been saved
    print("Saved MLTable file")

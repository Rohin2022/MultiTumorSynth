#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

def process_files(folder: Path):
    # List all file names in the given folder (non-recursively)
    file_names = [f.name for f in folder.iterdir() if f.is_file()]
    
    # Filter the list to keep only names that contain "BDMAP"
    file_names = [name for name in file_names if "BDMAP" in name]
    
    # Remove trailing "_gt.npz" or ".npz" from the names
    cleaned_names = []
    for name in file_names:
        if name.endswith("_gt.npz"):
            name = name[:-len("_gt.npz")]
        elif name.endswith(".npz"):
            name = name[:-len(".npz")]
        cleaned_names.append(name)
    
    # Remove duplicates
    unique_names = sorted(set(cleaned_names))
    return unique_names

def save_to_yaml(data, output_file: Path):
    # Save the list as a YAML file
    with output_file.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def main():
    parser = argparse.ArgumentParser(
        description="Process a folder of files to create a YAML list of dataset names"
    )
    parser.add_argument("folder", type=str, help="Folder path to process")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a valid directory.")
        return

    # Process the folder to extract unique names
    dataset_names = process_files(folder)
    
    # Create output directory "list" under the input folder
    output_dir = folder / "list"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset.yaml"

    # Save the list in YAML format
    save_to_yaml(dataset_names, output_file)

    print(f"Saved dataset list to {output_file}")

if __name__ == "__main__":
    main()
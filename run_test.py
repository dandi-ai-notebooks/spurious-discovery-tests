#!/usr/bin/env python3

import os
import sys
import shutil
from pathlib import Path
from minicline import perform_task

def main():
    if len(sys.argv) != 4:
        print("Usage: python process_dataset.py <dataset> <prompt> <model>")
        sys.exit(1)

    dataset = sys.argv[1]
    prompt = sys.argv[2]
    model = sys.argv[3]

    # Extract the last part of the model name after the '/'
    model_last_part = model.split('/')[-1]

    # Create working directory path
    working_dir = Path(f'tests/{dataset}/{prompt}/{model_last_part}/working')

    # Skip if directory already exists
    if working_dir.exists():
        print(f"Working directory {working_dir} already exists. Skipping.")
        sys.exit(0)

    # Create the directory and its parents
    working_dir.mkdir(parents=True)

    # Create .gitignore file
    with open(working_dir / '.gitignore', 'w') as f:
        f.write('data\n')

    # Copy specific files from dataset directory
    dataset_dir = Path(f'datasets/{dataset}')
    if not dataset_dir.exists():
        print(f"Dataset directory {dataset_dir} does not exist")
        sys.exit(1)

    # Copy only generate.py and readme.md
    for filename in ['generate.py', 'readme.md']:
        src_file = dataset_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, working_dir)

    # Change to working directory and execute generate.py
    original_cwd = os.getcwd()
    os.chdir(working_dir)
    os.system('python generate.py')
    os.chdir(original_cwd)

    # Remove generate.py from working directory
    (working_dir / 'generate.py').unlink()

    # Copy prompt instructions
    shutil.copy2(f'templates/{prompt}.txt', working_dir / 'instructions.txt')

    # Read instructions
    with open(working_dir / 'instructions.txt', 'r') as f:
        instructions = f.read()

    # Get absolute path to working directory
    full_working_dir = working_dir.absolute()

    # Set up log file path
    log_file = full_working_dir / 'minicline.log'

    # Perform the task
    perform_task(
        instructions=instructions,
        cwd=str(full_working_dir),
        model=model,
        log_file=str(log_file),
        auto=True,
        approve_all_commands=True
    )

if __name__ == '__main__':
    main()

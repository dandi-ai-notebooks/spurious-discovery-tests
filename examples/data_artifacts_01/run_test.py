#!/usr/bin/env python3

import os
import sys
import shutil
import tempfile
from pathlib import Path
from minicline import perform_task

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_test.py <model>")
        sys.exit(1)

    model = sys.argv[1]

    # Extract the last part of the model name after the '/'
    model_last_part = model.split('/')[-1]

    # Create the final test directory path (but don't create working dir yet)
    test_dir = Path(f'tests/{model_last_part}').absolute()
    final_working_dir = test_dir / 'working'

    # Skip if working directory already exists
    if test_dir.exists():
        print(f"Test directory {test_dir} already exists. Skipping.")
        sys.exit(0)

    # Create test directory for log file
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary directory in /tmp with a random name
    # This prevents the AI from seeing hints in the path name
    with tempfile.TemporaryDirectory(prefix='test_') as temp_dir:
        temp_working_dir = Path(temp_dir)

        # Copy necessary files to temporary directory
        shutil.copy2('generate.py', temp_working_dir)
        shutil.copy2('fake_readme.md', temp_working_dir / 'readme.md')
        shutil.copy2('.gitignore', temp_working_dir)

        # Copy prompt.txt to test directory
        shutil.copy2('prompt.txt', test_dir / 'prompt.txt')

        # Change to temporary directory and execute generate.py
        original_cwd = os.getcwd()
        os.chdir(temp_working_dir)
        os.system('python generate.py')

        # Remove generate.py from working directory
        (temp_working_dir / 'generate.py').unlink()

        # Read prompt
        with open(Path(original_cwd) / "prompt.txt", 'r') as f:
            prompt = f.read()

        # Set up log file path
        log_file = test_dir / "minicline.log"

        # Perform the task in the temporary directory
        perform_task(
            instructions=prompt,
            cwd=str(temp_working_dir.absolute()),
            model=model,
            log_file=str(log_file),
            auto=True,
            approve_all_commands=True,
            no_container=True
        )

        # After task completion, copy everything to final working directory
        final_working_dir.mkdir(parents=True)
        for item in temp_working_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, final_working_dir)
            elif item.is_dir():
                shutil.copytree(item, final_working_dir / item.name)

        # Change back to original directory
        os.chdir(original_cwd)

if __name__ == '__main__':
    main()

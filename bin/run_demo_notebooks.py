import glob
import os
import shutil
import subprocess
import sys
import tempfile

DEMO_URL = "https://github.com/ThirdAILabs/Demos.git"

IGNORED_NBS = [
    "DeployThirdaiwithDatabricks.ipynb",
]


def skip_nb(path):
    for ignored_nb in IGNORED_NBS:
        if ignored_nb in path:
            print(f"Skipping notebook: {path}")
            return True
    return False


def get_relative_notebook_paths(temp_dir):
    print(f"Cloning the repository from {DEMO_URL}...")
    subprocess.call(["git", "clone", DEMO_URL], cwd=temp_dir)
    print("Repository cloned successfully.")

    notebook_dir = os.path.join(temp_dir, "Demos", "**", "*.ipynb")
    print(f"Searching for notebooks in {notebook_dir}...")
    notebook_paths = glob.glob(notebook_dir, recursive=True)

    print(f"Found {len(notebook_paths)} notebooks. Applying filters...")
    notebook_paths = [path for path in notebook_paths if not skip_nb(path)]
    print(f"Number of notebooks after ignoring specified ones: {len(notebook_paths)}")

    notebook_paths = [path for path in notebook_paths if "msmarco" in path]
    print(f"Number of notebooks after filtering for 'msmarco': {len(notebook_paths)}")

    len_demos_dir = len(str(os.path.join(temp_dir, "Demos"))) + 1  # For slash
    relative_notebook_paths = [str(path)[len_demos_dir:] for path in notebook_paths]

    print("Cleaning up the temporary directory...")
    shutil.rmtree(temp_dir)
    print("Cleanup complete.")

    return relative_notebook_paths


def main():
    print("Starting the notebook execution process...")
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory at {temp_dir}")

    relative_notebook_paths = get_relative_notebook_paths(temp_dir)
    print(f"Total notebooks to run: {len(relative_notebook_paths)}")

    successes, failures = [], []
    for notebook_path in relative_notebook_paths:
        print(f"Running notebook: {notebook_path} in Docker container...")
        retcode = os.system(
            f'docker run -e "OPENAI_API_KEY=$OPENAI_API_KEY" -e "THIRDAI_KEY=$THIRDAI_KEY" thirdai/run_demos_build bash -c "python3 run_single_demo_notebook.py \'{notebook_path}\'"'
        )
        if retcode == 0:
            print(f"Notebook {notebook_path} executed successfully.")
            successes.append(notebook_path)
        else:
            print(f"Notebook {notebook_path} failed to execute.")
            failures.append(notebook_path)

    print("\nThe following notebooks have passed:")
    for success in successes:
        print(f"\t- {success}")

    print("\nThe following notebooks have failed:")
    for failure in failures:
        print(f"\t- {failure}")

    if failures:
        print("Some notebooks failed to execute, exiting with code 1.")
        sys.exit(1)
    else:
        print("All notebooks executed successfully, exiting with code 0.")


if __name__ == "__main__":
    main()

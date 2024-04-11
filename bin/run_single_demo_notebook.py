import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

DEMO_URL = "https://github.com/ThirdAILabs/Demos.git"


def get_notebook_path(temp_dir, relative_notebook_path):
    print(f"Cloning repository from {DEMO_URL} into {temp_dir}...")
    result = subprocess.call(["git", "clone", DEMO_URL], cwd=temp_dir)
    if result == 0:
        print("Repository cloned successfully.")
    else:
        print("Failed to clone the repository.")
    return os.path.join(temp_dir, "Demos", relative_notebook_path)


def run_demo_notebook(notebook_path):
    print(f"Preparing to run notebook at {notebook_path}...")
    if "msmarco" in notebook_path:
        with open(notebook_path) as notebook_file:
            print(f"Reading notebook file: {notebook_path}")
            nb_in = nbformat.read(notebook_file, as_version=4)
            working_dir = str(Path(notebook_path).parent)
            print(f"Notebook working directory set to: {working_dir}")

            try:
                print(f"Starting execution of the notebook: {notebook_path}")
                ep = ExecutePreprocessor(
                    timeout=None,
                    kernel_name="python3",
                    resources={"metadata": {"path": working_dir}},
                )
                ep.preprocess(nb_in)
                print(f"Notebook executed successfully: {notebook_path}")
                return 0
            except CellExecutionError as exec_error:
                print(f"Execution error in notebook: {notebook_path}:\n{exec_error}")
                for cell in nb_in.cells:
                    if cell.cell_type == "code":
                        for output in cell.get("outputs", []):
                            if output.output_type == "error":
                                print(
                                    f"Error in cell: {cell.source}\nError Message: {'; '.join(output.traceback)}"
                                )
                return 1
            except Exception as error:
                print(f"Unhandled exception in notebook: {notebook_path}:\n{error}")
                return 1
    else:
        print(f"Skipping {notebook_path}, does not match criteria for execution.")
        return 0


def main():
    print("Starting main process...")
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory created at {temp_dir}")

    relative_notebook_path = sys.argv[1]
    print(f"Relative notebook path provided: {relative_notebook_path}")

    notebook_path = get_notebook_path(temp_dir, relative_notebook_path)
    print(f"Full notebook path resolved to: {notebook_path}")

    exit_code = run_demo_notebook(notebook_path)

    print(f"Cleaning up temporary directory at {temp_dir}")
    shutil.rmtree(temp_dir)

    sys.exit(exit_code)


if __name__ == "__main__":
    print("Script execution started.")
    main()
    print("Script execution finished.")

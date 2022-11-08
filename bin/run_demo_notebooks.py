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


def get_notebook_paths(temp_dir):
    # Collect all of the jupyter notebooks in the Demos repo
    subprocess.call(["git", "clone", DEMO_URL], cwd=temp_dir)
    notebook_dir = os.path.join(temp_dir, "Demos", "*.ipynb")
    notebook_paths = glob.glob(notebook_dir)
    return notebook_paths


def run_demo_notebooks(notebook_paths, temp_dir):
    failed_notebooks = []
    for notebook_path in notebook_paths:
        if Path(notebook_path).stem == "FraudDetection":
            with open(notebook_path) as notebook_file:
                # Ref: https://nbformat.readthedocs.io/en/latest/format_description.html
                nb_in = nbformat.read(notebook_file, nbformat.NO_CONVERT)
                # The resources argument is needed to execute the notebook in the temporary directory
                temp_path = os.path.join(temp_dir, "Demos")
                try:
                    ep = ExecutePreprocessor(
                        timeout=None,
                        kernel_name="python3",
                        resources={"metadata": {"path": temp_path}},
                    )
                    nb_out = ep.preprocess(nb_in)
                except:
                    notebook_name = Path(notebook_path).stem
                    failed_notebooks.append(notebook_name)

    if failed_notebooks:
        sys.exit(f"The following notebooks failed due to error: {failed_notebooks}")
    else:
        print("All notebooks ran successfully")


def main():
    temp_dir = tempfile.mkdtemp()
    demo_notebook_paths = get_notebook_paths(temp_dir)
    run_demo_notebooks(demo_notebook_paths, temp_dir)
    shutil.rmtree(temp_dir)  # Clean up the files used for the test


if __name__ == "__main__":
    main()

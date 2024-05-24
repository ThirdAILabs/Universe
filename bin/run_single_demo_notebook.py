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
    # Collect all of the jupyter notebooks in the Demos repo
    subprocess.call(["git", "clone", DEMO_URL], cwd=temp_dir)
    subprocess.call(["git", "fetch"], cwd=f"{temp_dir}/Demos")
    subprocess.call(["git", "checkout", "fix-demos"], cwd=f"{temp_dir}/Demos")
    return os.path.join(temp_dir, "Demos", relative_notebook_path)


def run_demo_notebook(notebook_path):
    with open(notebook_path) as notebook_file:
        # Ref: https://nbformat.readthedocs.io/en/latest/format_description.html
        nb_in = nbformat.read(notebook_file, nbformat.NO_CONVERT)
        # The resources argument is needed to execute the notebook in a specific
        # directory. We run the notebooks in the directory they are located in
        # to ensure that paths work correctly to configs (or anything else).
        working_dir = str(Path(notebook_path).parent)
        try:
            print(f"Running notebook: {notebook_path}")
            ep = ExecutePreprocessor(
                timeout=None,
                kernel_name="python3",
                resources={"metadata": {"path": working_dir}},
            )
            nb_out = ep.preprocess(nb_in)
            print(f"Successfully ran the notebook: {notebook_path}")
            return 0
        except Exception as error:
            print(f"Failure in notebook: {notebook_path}:\n{error}")
            return 1


def main():
    temp_dir = tempfile.mkdtemp()
    relative_notebook_path = sys.argv[1]
    notebook_path = get_notebook_path(temp_dir, relative_notebook_path)
    exit_code = run_demo_notebook(notebook_path)
    shutil.rmtree(temp_dir)  # Clean up the files used for the test
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

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
    return os.path.join(temp_dir, "Demos", relative_notebook_path)


def run_demo_notebook(notebook_path):
    errors = []
    with open(notebook_path) as notebook_file:
        # Ref: https://nbformat.readthedocs.io/en/latest/format_description.html
        nb_in = nbformat.read(notebook_file, nbformat.NO_CONVERT)
        # The resources argument is needed to execute the notebook in a specific
        # directory. We run the notebooks in the directory they are located in
        # to ensure that paths work correctly to configs (or anything else).
        working_dir = str(Path(notebook_path).parent)
        try:
            ep = ExecutePreprocessor(
                timeout=None,
                kernel_name="python3",
                resources={"metadata": {"path": working_dir}},
            )
            nb_out = ep.preprocess(nb_in)
        except Exception as error:
            notebook_name = Path(notebook_path).stem
            errors.append((notebook_name, error))

    if errors:
        for failed_notebook, error in errors:
            print(f"Failure in notebook: {failed_notebook}: \n {error}")
        sys.exit(1)
    else:
        print("Successfully ran the following notebooks:")
        print(f"\t{Path(notebook_path).stem}")


def main():
    temp_dir = tempfile.mkdtemp()
    relative_notebook_path = sys.argv[1]
    notebook_path = get_notebook_path(temp_dir, relative_notebook_path)
    run_demo_notebook(notebook_path)
    shutil.rmtree(temp_dir)  # Clean up the files used for the test


if __name__ == "__main__":
    main()

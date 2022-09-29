import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

pytestmark = [pytest.mark.unit]

TIMEOUT = 600
DEMO_URL = "https://github.com/ThirdAILabs/Demos.git" 

@pytest.mark.release
class TestDemoNotebooks():
    def setup_class(self):
        # Collect all of the jupyter notebooks in the Demos repo
        self.temp_dir = tempfile.mkdtemp()
        subprocess.call(["git", "clone", DEMO_URL], cwd=self.temp_dir)
        notebook_dir = os.path.join(self.temp_dir, "Demos", '*.ipynb')
        self.notebook_paths = glob.glob(notebook_dir)

    def teardown_class(self):
        shutil.rmtree(self.temp_dir)

    def test_demo_notebooks(self):
        for notebook_path in self.notebook_paths:
            with open(notebook_path) as notebook_file:
                nb_in = nbformat.read(notebook_file, nbformat.NO_CONVERT)
            try:
                # The resources argument is needed to execute the notebook in the temporary directory
                ep = ExecutePreprocessor(timeout=TIMEOUT, kernel_name='python3', resources={'metadata': {'path': self.temp_dir}})
                nb_out = ep.preprocess(nb_in)
            except CellExecutionError:
                notebook_name = Path(notebook_path).stem
                pytest.fail(f"The notebook {notebook_name} failed when executed")


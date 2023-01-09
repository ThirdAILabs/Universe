# These tests use go, but to make it easy to fit into our current framework
# they are wrapped in a test runnable by pytest

import os
import pathlib
import subprocess

import pytest


@pytest.mark.unit
def test_go_server():
    go_src_directory = pathlib.Path(__file__).parent.parent / "src" / "server"
    os.chdir(go_src_directory)
    assert subprocess.run(f"go test", shell=True).returncode == 0

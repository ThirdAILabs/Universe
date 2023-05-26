# These tests use go, but to make it easy to fit into our current framework
# they are wrapped in a test runnable by pytest

import os
import pathlib
import subprocess

import pytest


@pytest.fixture
def switch_to_server_dir():
    original_directory = os.getcwd()

    go_src_directory = (
        pathlib.Path(__file__).parent.parent / "src" / "methods" / "server"
    )

    # Change to go package directory
    os.chdir(go_src_directory)

    yield

    # Return to original working directory
    os.chdir(original_directory)


@pytest.mark.unit
def test_go_server(switch_to_server_dir):
    assert subprocess.run(f"go test", shell=True).returncode == 0

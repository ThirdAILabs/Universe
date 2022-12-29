
import pytest
from licensing_utils import this_should_require_a_license_bolt
import time

pytestmark = [pytest.mark.release]

from pathlib import Path
import subprocess
import os

import thirdai


invalid_local_port = "97531"
valid_local_port = "8080"

invalid_heartbeat_location = f"http://localhost:{invalid_local_port}"
valid_heartbeat_location = f"http://localhost:{valid_local_port}"

max_num_workers = 8

dir_path = Path(__file__).resolve().parent
go_build_script = dir_path / ".." / "bin" / "build_license_server.py"
go_run_script = dir_path / ".." / "src" / "server" / f"license-server-max-{max_num_workers}"


def setup_module():
    build_command = f"{go_build_script.resolve()} {max_num_workers}"
    os.system(build_command)

@pytest.fixture()
def startup_license_server():
    server_process = subprocess.Popen(str(go_run_script.resolve()), shell=True)
    time.sleep(2) # TODO: Replace this with listening for server started output
    yield
    server_process.kill()

def test_with_invalid_heartbeat_location():

    with pytest.raises(
        RuntimeError,
        match=f"Could not establish initial connection to licensing server.",
    ):
        thirdai.start_heartbeat(invalid_heartbeat_location)


def test_valid_heartbeat(startup_license_server):
    thirdai.start_heartbeat(valid_heartbeat_location)

# def test_heartbeat_multiple_machines(startup_license_server):
#     pass

# def test_heartbeat_too_many_machines(startup_license_server):
#     pass

# def test_heartbeat_timeout(startup_license_server):
#     pass

# def test_more_machines_after_timeout(startup_license_server):
#     pass
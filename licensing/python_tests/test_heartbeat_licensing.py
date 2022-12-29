import os
import subprocess
import time
from pathlib import Path

import pytest
import requests
import thirdai
from licensing_utils import this_should_require_a_license_bolt

pytestmark = [pytest.mark.release]

invalid_local_port = "97531"
valid_local_port = "8080"

invalid_heartbeat_location = f"http://localhost:{invalid_local_port}"
valid_heartbeat_location = f"http://localhost:{valid_local_port}"

max_num_workers = 8

dir_path = Path(__file__).resolve().parent
go_build_script = dir_path / ".." / "bin" / "build_license_server.py"
go_run_script = (
    dir_path / ".." / "src" / "server" / f"license-server-max-{max_num_workers}"
)


def setup_module():
    build_command = f"{go_build_script.resolve()} {max_num_workers}"
    os.system(build_command)


def wait_for_server_start():
    max_wait_time_seconds = 10
    retry_period_seconds = 0.25
    start_time_seconds = time.time()
    while True:
        try:
            requests.get(valid_heartbeat_location)
            return
        except requests.exceptions.ConnectionError:
            if time.time() - start_time_seconds > max_wait_time_seconds:
                raise RuntimeError("License server took too long to start")
            time.sleep(retry_period_seconds)


def wait_for_server_end():
    max_wait_time_seconds = 10
    retry_period_seconds = 0.25
    start_time_seconds = time.time()
    while True:
        try:
            requests.get(valid_heartbeat_location)
            if time.time() - start_time_seconds > max_wait_time_seconds:
                raise RuntimeError("License server took too long to start")
            time.sleep(retry_period_seconds)
        except requests.exceptions.ConnectionError:
            return


@pytest.fixture()
def license_server():
    server_process = subprocess.Popen(
        str(go_run_script.resolve()), stdout=subprocess.PIPE, universal_newlines=True
    )
    wait_for_server_start()
    yield
    server_process.kill()
    wait_for_server_end()


def test_with_invalid_heartbeat_location():
    with pytest.raises(
        RuntimeError,
        match=f"Could not establish initial connection to licensing server.",
    ):
        thirdai.start_heartbeat(invalid_heartbeat_location)


def test_valid_heartbeat(license_server):
    thirdai.start_heartbeat(valid_heartbeat_location)
    this_should_require_a_license_bolt()


# def test_heartbeat_multiple_machines(startup_license_server):
#     pass

# def test_heartbeat_too_many_machines(startup_license_server):
#     pass

# def test_heartbeat_timeout(startup_license_server):
#     pass

# def test_more_machines_after_timeout(startup_license_server):
#     pass

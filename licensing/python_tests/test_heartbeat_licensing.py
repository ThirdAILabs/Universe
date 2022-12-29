import os
import subprocess
import time
from pathlib import Path

import pytest
import requests
import thirdai
from licensing_utils import LOCAL_HEARTBEAT_SERVER, this_should_require_a_license_bolt

pytestmark = [pytest.mark.release]

invalid_local_port = "97531"
valid_local_port = "8080"

invalid_heartbeat_location = f"http://localhost:97531"

max_num_workers = 3

python_test_dir_path = Path(__file__).resolve().parent
go_build_script = python_test_dir_path / ".." / "bin" / "build_license_server.py"
go_run_script = (
    python_test_dir_path
    / ".."
    / "src"
    / "server"
    / f"license-server-max-{max_num_workers}"
)
heartbeat_script = python_test_dir_path / "heartbeat_script.py"


def setup_module():
    build_command = f"{go_build_script.resolve()} {max_num_workers}"
    os.system(build_command)


def wait_for_server_start():
    max_wait_time_seconds = 10
    retry_period_seconds = 0.25
    start_time_seconds = time.time()
    while True:
        try:
            requests.get(LOCAL_HEARTBEAT_SERVER)
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
            requests.get(LOCAL_HEARTBEAT_SERVER)
            if time.time() - start_time_seconds > max_wait_time_seconds:
                raise RuntimeError("License server took too long to start")
            time.sleep(retry_period_seconds)
        except requests.exceptions.ConnectionError:
            return


@pytest.fixture(scope="function")
def license_server():
    server_process = subprocess.Popen(
        str(go_run_script.resolve()), stdout=subprocess.PIPE, universal_newlines=True
    )
    wait_for_server_start()
    yield server_process
    server_process.kill()
    wait_for_server_end()


def test_with_invalid_heartbeat_location():
    with pytest.raises(
        RuntimeError,
        match=f"Could not establish initial connection to licensing server.",
    ):
        thirdai.start_heartbeat(invalid_heartbeat_location)


def test_with_invalid_heartbeat_grace_period():
    with pytest.raises(
        ValueError,
        match=f"Heartbeat timeout must be less than 10000 seconds.",
    ):
        thirdai.start_heartbeat(invalid_heartbeat_location, heartbeat_timeout=100000)


def test_valid_heartbeat(license_server):
    thirdai.start_heartbeat(LOCAL_HEARTBEAT_SERVER)
    this_should_require_a_license_bolt()
    thirdai.end_heartbeat()


def test_heartbeat_multiple_machines(license_server):

    for _ in range(max_num_workers):
        assert (
            subprocess.run(
                f"python3 {heartbeat_script.resolve()}", shell=True
            ).returncode
            == 0
        )

    assert (
        subprocess.run(f"python3 {heartbeat_script.resolve()}", shell=True).returncode
        != 0
    )


# def test_more_machines_after_server_timeout(fast_timeout_license_server):
#     pass


def test_client_side_timeout_after_heartbeat_fail(license_server):
    thirdai.start_heartbeat(LOCAL_HEARTBEAT_SERVER, heartbeat_timeout=1)
    this_should_require_a_license_bolt()
    license_server.kill()
    wait_for_server_end()
    # Sleep for 3 seconds to ensure that the heartbeat (which is once a second)
    # runs at least once one second after the server goes down
    time.sleep(3)
    with pytest.raises(
        RuntimeError,
        match=f"The heartbeat thread could not verify with the server because there has not been a successful heartbeat in 1 seconds.*",
    ):
        this_should_require_a_license_bolt()

    # TODO(Josh): Add metrics check to this

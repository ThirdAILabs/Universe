import os
import subprocess
import time
from pathlib import Path

import pytest
import requests
import thirdai
from licensing_utils import (
    LOCAL_HEARTBEAT_SERVER,
    deactivate_license_at_start_of_demo_test,
    run_udt_training_routine,
)

pytestmark = [pytest.mark.release]

invalid_heartbeat_location = f"http://localhost:97531"

# Since the name of the license server executable is just augmented with the max
# number of machines and not the machine timeout, we use a different number of
# workers for the "normal" server and the "fast timeout" server.
max_num_workers_normal = 3

# The fast timeout info here describes a licensing server where the timeout for
# when machines are considered no longer active is very low. This allows us to
# test timeout behavior without waiting the usual amount of time necessary for
# the server to consider machines no longer active.
max_num_workers_fast_timeout = 1
fast_timeout_ms = 10

python_test_dir_path = Path(__file__).resolve().parent
go_build_script = python_test_dir_path / ".." / "bin" / "build_license_server.py"
heartbeat_script = python_test_dir_path / "heartbeat_script.py"


def setup_module():
    normal_build_command = f"{go_build_script.resolve()} {max_num_workers_normal}"
    os.system(normal_build_command)

    fast_timeout_server_build_command = f"{go_build_script.resolve()} {max_num_workers_fast_timeout} --machine_timeout_ms {fast_timeout_ms}"
    os.system(fast_timeout_server_build_command)


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
                raise RuntimeError("License server took too long to end")
            time.sleep(retry_period_seconds)
        except requests.exceptions.ConnectionError:
            return


def license_server_helper(max_workers, do_not_sign_responses=False):
    go_run_script = [
        str(
            (
                python_test_dir_path
                / ".."
                / "src"
                / "methods"
                / "server"
                / f"license-server-max-{max_workers}"
            ).resolve()
        )
    ]
    if do_not_sign_responses:
        go_run_script.append("--do_not_sign_responses")
    server_process = subprocess.Popen(
        go_run_script, stdout=subprocess.PIPE, universal_newlines=True
    )
    wait_for_server_start()
    yield server_process
    server_process.kill()
    wait_for_server_end()


@pytest.fixture(scope="function")
def normal_license_server():
    yield from license_server_helper(max_num_workers_normal)


@pytest.fixture(scope="function")
def no_signing_license_server():
    yield from license_server_helper(max_num_workers_normal, do_not_sign_responses=True)


@pytest.fixture(scope="function")
def fast_timeout_license_server():
    yield from license_server_helper(max_num_workers_fast_timeout)


def test_with_invalid_heartbeat_location():
    with pytest.raises(
        RuntimeError,
        match=f"Could not establish initial connection to licensing server.",
    ):
        thirdai.licensing.start_heartbeat(invalid_heartbeat_location)
    thirdai.licensing.deactivate()


def test_with_invalid_heartbeat_grace_period():
    with pytest.raises(
        ValueError,
        match=f"Heartbeat timeout must be less than 10000 seconds.",
    ):
        thirdai.licensing.start_heartbeat(
            invalid_heartbeat_location, heartbeat_timeout=100000
        )
    thirdai.licensing.deactivate()


def test_heartbeat_fails_with_no_signature(no_signing_license_server):
    with pytest.raises(
        RuntimeError,
        match=f"Could not establish initial connection to licensing server.",
    ):
        thirdai.licensing.start_heartbeat(LOCAL_HEARTBEAT_SERVER)
    thirdai.licensing.deactivate()


def test_valid_heartbeat(normal_license_server):
    thirdai.licensing.start_heartbeat(LOCAL_HEARTBEAT_SERVER)
    run_udt_training_routine()
    thirdai.licensing.deactivate()


def test_heartbeat_multiple_machines(normal_license_server):
    for _ in range(max_num_workers_normal):
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
    thirdai.licensing.deactivate()


def test_more_machines_after_server_timeout(fast_timeout_license_server):
    for _ in range(max_num_workers_fast_timeout * 3):
        assert (
            subprocess.run(
                f"python3 {heartbeat_script.resolve()}", shell=True
            ).returncode
            == 0
        )
        time.sleep(fast_timeout_ms / 1000)


def test_client_side_timeout_after_heartbeat_fail(normal_license_server):
    thirdai.licensing.start_heartbeat(LOCAL_HEARTBEAT_SERVER, heartbeat_timeout=1)
    run_udt_training_routine()
    normal_license_server.kill()
    wait_for_server_end()
    # Sleep for 3 seconds to ensure that the heartbeat (which is once a second)
    # runs at least once one second after the server goes down
    time.sleep(3)
    with pytest.raises(
        RuntimeError,
        match=f"The heartbeat thread could not verify with the server because there has not been a successful heartbeat in 1 seconds.*",
    ):
        run_udt_training_routine()
    thirdai.licensing.deactivate()


def test_maintenance_of_valid_heartbeat(normal_license_server):
    """
    This test tests standard heartbeats (as opposed to just the original
    heartbeat) by using a heartbeat_timeout of 0 seconds, which means that
    verification will only work if the last heartbeat worked. We sleep
    for a second to ensure that the heartbeat has started and that the most
    recent heartbeat call was not the original one.
    """
    thirdai.licensing.start_heartbeat(LOCAL_HEARTBEAT_SERVER, heartbeat_timeout=0)
    time.sleep(1)
    run_udt_training_routine()
    thirdai.licensing.deactivate()

import atexit
import pathlib
import subprocess
import sys
import uuid
from typing import Optional

import thirdai

daemon_path = pathlib.Path(__file__).parent.resolve() / "telemetry_daemon.py"

background_process = None

BACKGROUND_THREAD_TIMEOUT_SECONDS = 0.5

UUID = uuid.uuid4().hex


def kill_background_process():
    global background_process
    if background_process != None:
        poll = background_process.poll()
        if poll is not None:
            raise ValueError(
                f"Telemetry process terminated early with exit code {poll}"
            )
        background_process.terminate()
        try:
            background_process.wait(timeout=BACKGROUND_THREAD_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            background_process.kill()
    background_process = None


atexit.register(kill_background_process)


wrapped_start_method = thirdai._thirdai.telemetry.start
wrapped_stop_method = thirdai._thirdai.telemetry.stop


def start(
    port: Optional[int] = None,
    write_dir: Optional[str] = None,
    optional_endpoint_url: Optional[str] = None,
):
    global background_process
    if background_process != None:
        raise RuntimeError(
            "Trying to start telemetry client when one is already running"
        )

    if port:
        telemetry_url = wrapped_start_method(port)
    else:
        telemetry_url = wrapped_start_method()

    if write_dir == None:
        return telemetry_url

    # Could also try using os.fork
    python_executable = sys.executable
    push_location = write_dir + f"/telemetry-" + UUID
    background_process = subprocess.Popen(
        [
            python_executable,
            str(daemon_path.resolve()),
            "--telemetry_url",
            telemetry_url,
            "--push_location",
            push_location,
            "--optional_endpoint_url",
            str(optional_endpoint_url),
        ]
    )

    return push_location


def stop():
    kill_background_process()
    wrapped_stop_method()

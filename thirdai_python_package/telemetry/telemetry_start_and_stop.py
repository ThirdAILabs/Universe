import atexit
import pathlib
import subprocess
import sys
from typing import Optional

import pandas as pd
import thirdai

daemon_path = pathlib.Path(__file__).parent.resolve() / "telemetry_daemon.py"

background_process = None

BACKGROUND_THREAD_TIMEOUT_SECONDS = 0.5


def kill_background_process():
    global background_process
    if background_process != None:
        background_process.terminate()
        try:
            background_process.wait(timeout=BACKGROUND_THREAD_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            background_process.kill()
    background_process = None


atexit.register(kill_background_process)


wrapped_start_method = thirdai._thirdai.telemetry.start
wrapped_stop_method = thirdai._thirdai.telemetry.stop


def start(port: Optional[int] = None, write_location: Optional[str] = None):
    global background_process
    if background_process != None:
        raise RuntimeError(
            "Trying to start telemetry client when one is already running"
        )

    if port:
        url = wrapped_start_method(port)
    else:
        url = wrapped_start_method()

    if write_location != None:
        # Could also try using os.fork
        python_executable = sys.executable
        background_process = subprocess.Popen(
            [
                python_executable,
                str(daemon_path.resolve()),
                "--telemetry_url",
                "http://" + url + "/metrics",
                "--push_location",
                write_location,
            ]
        )


def stop():
    kill_background_process()
    wrapped_stop_method()

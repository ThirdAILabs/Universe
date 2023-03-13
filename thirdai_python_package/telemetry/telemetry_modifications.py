import atexit
import pathlib
import subprocess
import sys
from typing import Optional

import pandas as pd
import thirdai._thirdai.telemetry as telemetry

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


def modified_telemetry_start():
    original_start_method = telemetry.start
    original_stop_method = telemetry.stop

    def wrapped_start(
        self, push_location: Optional[str] = None, port: Optional[int] = None
    ):
        global background_process
        if background_process != None:
            raise RuntimeError(
                "Trying to start telemetry client when one is already running"
            )

        if port:
            url = original_start_method(port)
        else:
            url = original_start_method()

        if push_location:
            # Could also try using os.fork
            python_executable = sys.executable
            background_process = subprocess.POpen(
                [
                    python_executable,
                    daemon_path,
                    "--telemetry_url",
                    url,
                    "--push_location",
                    push_location,
                ]
            )

    def wrapped_stop(self):
        kill_background_process()
        telemetry.stop()

    delattr(telemetry, "start")
    delattr(telemetry, "stop")

    telemetry.start = wrapped_start
    telemetry.stop = wrapped_stop

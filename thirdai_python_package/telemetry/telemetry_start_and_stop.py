import atexit
import pathlib
import subprocess
import sys
import time
from typing import Optional

from thirdai._thirdai import telemetry

daemon_path = pathlib.Path(__file__).parent.resolve() / "telemetry_daemon.py"

background_telemetry_push_process = None

# If we have to wait longer than this many seconds when trying to gracefully
# terminate the background process, we give up and send a kill message, possibly
# losing telemetry data.
BACKGROUND_THREAD_TIMEOUT_SECONDS = 0.5

# Wait this many seconds after starting the background thread before checking if
# it is still running.
BACKGROUND_THREAD_HEALTH_CHECK_WAIT = 0.5


def _assert_background_telemetry_push_process_running():
    poll = background_telemetry_push_process.poll()

    if poll is not None:
        raise ValueError(f"Telemetry process terminated early with exit code {poll}")


# See https://stackoverflow.com/q/320232/ensuring-subprocesses-are-dead-on-exiting-python-program
# If a background telemetry push process (as started by a call to start) exists,
# this function tries to gracefully kill that process by sending a sigkill.
# If the process doesn't finish quickly enough, this function sends a sigterm,
# which will force kill it.
def _kill_background_telemetry_push_process():
    global background_telemetry_push_process
    if background_telemetry_push_process != None:
        _assert_background_telemetry_push_process_running()
        background_telemetry_push_process.terminate()
        try:
            background_telemetry_push_process.wait(
                timeout=BACKGROUND_THREAD_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            background_telemetry_push_process.kill()

    background_telemetry_push_process = None


# This will cause _kill_background_telemetry_push_process to get called when
# the current interpreter session finishes. According to
# https://docs.python.org/3/library/atexit.html, this is not called when the
# program is killed by a signal not handled by Python, when a Python fatal
# internal error is detected, or when os._exit() is called.
atexit.register(_kill_background_telemetry_push_process)


wrapped_start_method = telemetry.start
wrapped_stop_method = telemetry.stop


def start(
    port: Optional[int] = None,
    write_dir: Optional[str] = None,
    optional_endpoint_url: Optional[str] = None,
):
    global background_telemetry_push_process
    if background_telemetry_push_process != None:
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
    args = [
        python_executable,
        str(daemon_path.resolve()),
        "--telemetry_url",
        telemetry_url,
        "--push_dir",
        write_dir,
        "--optional_endpoint_url",
        str(optional_endpoint_url),
    ]
    background_telemetry_push_process = subprocess.Popen(args)

    time.sleep(BACKGROUND_THREAD_HEALTH_CHECK_WAIT)
    _assert_background_telemetry_push_process_running()

    push_location = write_dir + f"/telemetry-" + telemetry.uuid()
    return push_location


def stop():
    _kill_background_telemetry_push_process()
    wrapped_stop_method()

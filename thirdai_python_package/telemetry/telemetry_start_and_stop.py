import atexit
import pathlib
import subprocess
import sys
import time
from typing import Optional

from thirdai._thirdai import telemetry

from .telemetry_daemon import push_telemetry

# This file defines methods start and stop, which get added to the
# thirdai.telemetry module in __init__.py. They wrap the corresponding start and
# stop methods in thirdai._thirdai.telemetry, which start serving the prometheus
# metrics on the passed in port. The reason we need to wrap these methods is so
# that we can additionally start and stop a background thread that *pushes*
# prometheus metrics from the local Prometheus endpoint (localhost:port) to a
# file/remote endpoint. See telemetry_daemon.py for the background daemon that
# gets started. The push location can be an s3 path or a file path, and the
# daemon will handle pushing to the correct location.
#
# telemetry.start() should be called by the user at the top of a script that
# they want to track telemetry in. telemetry.stop() mostly should never need to
# be called by the user.
#
# We cannot use a python thread for the background push daemon because of the
# GIL. We cannot easily use a C++ background thread because of the complexity
# of writing e.g. S3 adapters in C++.


daemon_script_path = pathlib.Path(__file__).parent.resolve() / "telemetry_daemon.py"

background_telemetry_push_process = None

# If we have to wait longer than this many seconds when trying to gracefully
# terminate the background process, we give up and send a kill message, possibly
# losing telemetry data.
BACKGROUND_THREAD_TIMEOUT_SECONDS = 2

# Wait this many seconds after starting the background thread before checking if
# it is still running.
BACKGROUND_THREAD_HEALTH_CHECK_WAIT = 0.5

# We will upload data to the push dir at this interval (and before the script
# finishes when the GracefulKiller catches that exception)
DEFAULT_UPLOAD_INTERVAL_SECONDS = 60 * 20


# See https://stackoverflow.com/q/320232/ensuring-subprocesses-are-dead-on-exiting-python-program
# If a background telemetry push process (as started by a call to start) exists,
# this function tries to gracefully kill that process by sending a sigkill.
# If the process doesn't finish quickly enough, this function sends a sigterm,
# which will force kill it.
def _kill_background_telemetry_push_process():
    global background_telemetry_push_process
    if (
        background_telemetry_push_process != None
        and background_telemetry_push_process.poll() is None
    ):
        background_telemetry_push_process.terminate()
        try:
            background_telemetry_push_process.wait(
                timeout=BACKGROUND_THREAD_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            background_telemetry_push_process.kill()

    background_telemetry_push_process = None


# This will cause _kill_background_telemetry_push_process to get called when
# the current interpreter session finishes, which will kill the background
# process and cause it to do one last push. According to
# https://docs.python.org/3/library/atexit.html, this is not called when the
# program is killed by a signal not handled by Python, when a Python fatal
# internal error is detected, or when os._exit() is called. Since this is
# defined after we define _thirdai, it should get called before the destructors
# that take down the metrics server.
atexit.register(_kill_background_telemetry_push_process)


wrapped_start_method = telemetry.start
wrapped_stop_method = telemetry.stop


def start(
    port: Optional[int] = None,
    write_dir: Optional[str] = None,
    write_period_seconds: int = DEFAULT_UPLOAD_INTERVAL_SECONDS,
    optional_endpoint_url: Optional[str] = None,
):
    """
    Start a Prometheus telemetry client on the passed in port. If a port is not
    specified this method will use the default ThirdAI port of 9929. This
    function is not thread safe with other ThirdAI code, so you should make sure
    that no other code is running when this method is called.
    If a write_dir is passed in, this function will additionally start a
    background daemon that will push the Prometheus telemetry to write_dir at
    the path write_dir/telemetry-<instance_uuid>. Currently, write_dir can be a
    local path or an s3 path. Writes to this write_dir will happen every
    write_period_seconds, which has a default of 20 minutes.
    """
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

    # Run serially once to fix most errors
    push_telemetry(write_dir, telemetry_url, optional_endpoint_url)

    # Could also try using os.fork
    python_executable = sys.executable
    args = [
        python_executable,
        str(daemon_script_path.resolve()),
        "--telemetry_url",
        telemetry_url,
        "--push_dir",
        write_dir,
        "--upload_interval_seconds",
        str(write_period_seconds),
    ]
    if optional_endpoint_url:
        args += ["--optional_endpoint_url", optional_endpoint_url]
    background_telemetry_push_process = subprocess.Popen(args)

    push_location = write_dir + f"/telemetry-" + telemetry.uuid()
    return push_location


def stop():
    """
    Stops the current Prometheus telemetry client if one is running. This
    function is not thread safe with other ThirdAI code, so you should make sure
    that no other code is running when this method is called.
    """
    _kill_background_telemetry_push_process()
    wrapped_stop_method()

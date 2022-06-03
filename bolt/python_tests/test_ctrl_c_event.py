import signal
import subprocess
import time
import os
from subprocess import DEVNULL, PIPE
import sys
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.release]


@pytest.mark.unit
def test_ctrl_c_functionality():
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin')::
        # started a subprocesss to run a file to which SIGINT needed to be sent
        proc = subprocess.Popen(
            ["python3", "ctrl_c_executable.py"],
            shell=False,
            stdout=PIPE,
            stderr=DEVNULL,
        )

        # making sure the program reaches the train function
        for output_from_proc in proc.stdout:
            output_from_proc_decode = output_from_proc.decode("utf-8")
            if (
                output_from_proc_decode.find(
                    "Marker to check train function about to start"
                )
                != -1
            ):
                break
        time.sleep(1)
        # sending the SIGINT Signal to the subprocess
        proc.send_signal(signal.SIGINT)
        time.sleep(2)
        # checking the status of the process, if it is still running,
        # means that the SIGINT is not handled by the process, else
        # it should be handled by the process
        poll = proc.poll()
        proc.kill()
        proc.wait()

        if poll is None:
            assert False, f"CTRL+C Functionality not working correctly"

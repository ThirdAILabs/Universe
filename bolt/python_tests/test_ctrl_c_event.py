import signal
import subprocess
import time
import os
from subprocess import DEVNULL, PIPE
import sys
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.release]


def test_ctrl_c_functionality():
    # windows could not pass the test,hence running only on linux and darwin
    if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
        # started a subprocesss to run a file to which SIGINT needed to be sent
        proc = subprocess.Popen(
            ["python3", "../bolt/python_tests/ctrl_c_executable.py"],
            stdout=PIPE,
            stderr=DEVNULL,
            shell=False,
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

        # The sleep below is to wait till the train functtion start executing
        # For making sure the program doesn't terminate before epoches for
        # the train function is 10000000
        time.sleep(1)
        proc.send_signal(signal.SIGINT)

        # The sleep below is to wait till the SIGINT is received by the subprocess
        # and the subprocess is terminated For making sure the program doesn't
        # terminate before sleep() returns, epoches for the train function is 10000000
        time.sleep(2)

        # poll returns the exitcode if the process has terminated and otherwise returns none. We can check
        # if the program terminated correctly with a SIGINT by asserting that the result of a poll is the
        # SIGINT signal code. We also need to kill the process and wait, just in case this test fails,
        # since we don't want the subprocess to keep on running after the program ends.
        poll = proc.poll()
        proc.kill()
        proc.wait()
        assert poll == signal.SIGINT, f"CTRL+C Functionality not working correctly"

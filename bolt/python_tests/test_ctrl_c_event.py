import signal
import subprocess
import time
import os
from subprocess import DEVNULL
import pytest


@pytest.mark.unit
def test_ctrl_c_functionality():
    # started a subprocesss to run a file to which SIGINT needed to be sent
    proc = subprocess.Popen(
        ["python3", "../bolt/python_tests/ctrl_c_executable.py"],
        shell=False,
        stdout=DEVNULL,
        stderr=DEVNULL,
    )

    # making sure the program reaches the train function
    time.sleep(3)
    # sending the SIGINT Signal to the subprocess

    proc.send_signal(signal.SIGINT)
    time.sleep(2)
    # checking the status of the process, if it is still running,
    # means that the SIGINT is not handled by the process, else
    # it should be handled by the process
    poll = proc.poll()
    if poll is None:
        proc.kill()
        assert False, f"CTRL+C Functionality not working correctly"
    elif poll < 0:
        proc.kill()
        assert True
    else:
        assert True


test_ctrl_c_functionality()

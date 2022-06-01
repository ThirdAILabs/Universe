import signal
import subprocess
import time
import os
import pytest
from subprocess import Popen, PIPE, STDOUT
MAX_CALLS = 10 #running the scripts multiple time to check it works on every instance

@pytest.mark.unit
def test_ctrl_c_functionality():
  proc = subprocess.Popen(['python3', '../bolt/python_tests/ctrl_c_executable.py'] ,
                          shell=False,  stdout = PIPE)
  try:
    time.sleep(3)
    proc.send_signal(signal.SIGINT)
    time.sleep(1)
  except:
    assert True
  else:
    poll = proc.poll()
    if poll is None:
      proc.kill();
      assert False, f"CTRL+C Functionality not working correctly"
    


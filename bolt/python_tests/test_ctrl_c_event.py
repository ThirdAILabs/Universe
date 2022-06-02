import signal
import subprocess
import time
import os
import pytest
from subprocess import Popen, DEVNULL
MAX_CALLS = 10 #running the scripts multiple time to check it works on every instance

@pytest.mark.unit
def test_ctrl_c_functionality():
  proc = subprocess.Popen(['python3', "-W", "ignore", '../bolt/python_tests/ctrl_c_executable.py'] ,
                          shell=False,  stdout = DEVNULL, stderr= DEVNULL)
  try:
    time.sleep(3)
    proc.send_signal(signal.SIGINT)
    time.sleep(2)
  except:
    assert True
  else:
    poll = proc.poll()
    if poll is None:
      proc.kill();
      assert False, f"CTRL+C Functionality not working correctly"
  
  


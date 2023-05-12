import json
import subprocess
import os


curr_path = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call(["bash", os.path.join(curr_path, "test_bash.sh")])
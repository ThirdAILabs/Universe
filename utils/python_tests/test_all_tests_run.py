import io
import os
import pathlib
from contextlib import redirect_stdout

import pytest


# This test collects all of the python tests that we run in github actions
# (currently unit, integration, release, and distributed, as well as some we
# explicitly don't run marked with ignore) and ensures that they
# cover all python tests we have overall.
# We just need to hope that THIS test always runs!
@pytest.mark.unit
def test_all_tests_run():
    universe_path = pathlib.Path(__file__).parent.parent.parent.resolve()
    os.chdir(universe_path)

    all_tests_buffer = io.StringIO()
    with redirect_stdout(all_tests_buffer):
        pytest.main([".", "--ignore-glob=deps", "--collect-only"])
    all_tests = [line.strip() for line in all_tests_buffer.getvalue().split("\n")]

    tests_we_run_buffer = io.StringIO()
    with redirect_stdout(tests_we_run_buffer):
        for run in ["unit", "release", "distributed", "ignore", "skip"]:
            pytest.main([".", "--ignore-glob=deps", "--collect-only", f"-m {run}"])
    tests_we_run = [line.strip() for line in tests_we_run_buffer.getvalue().split("\n")]

    fail = False
    for line in all_tests:
        if line.startswith("<Function") and line not in tests_we_run:
            print(f"We are not running {line}")
            fail = True

    assert not fail

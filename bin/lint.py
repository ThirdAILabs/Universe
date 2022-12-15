#!/usr/bin/env python3

import multiprocessing
import os
import subprocess
import time
from pathlib import Path


def get_our_files(patterns):
    for pattern in patterns:
        for header in Path(".").rglob(pattern):
            header = str(header.resolve())
            if "/deps/" in header or "/build/" in header:
                continue
            yield header


def pragma_once_check():
    files_that_need_pragma = []
    for header in get_our_files(["*.h"]):
        with open(header) as f:
            if f.readline().strip() != "#pragma once":
                files_that_need_pragma.append(header)

    if len(files_that_need_pragma) > 0:
        print("The following files need pragma once: " + str(files_that_need_pragma))
        exit(1)


def ensure_compile_commands_db_created():
    subprocess.run(["bin/generate_compile_commands.sh"])


def universe_dir():
    return Path(__file__).parent.parent


def run_clang_tidy():
    # Maximum number of concurrent jobs to run
    max_concurrent = int(1.5 * multiprocessing.cpu_count())
    # Contains a list of (command, log_path) pairs
    commmands_to_run = []

    for cc_file in get_our_files(["*.cc"]):
        from pathlib import Path

        log_file_location = Path("clang_tidy_logs") / Path(cc_file).relative_to(
            universe_dir()
        ).with_suffix(".log")
        log_file_location.parent.mkdir(parents=True, exist_ok=True)
        log_file_location = str(log_file_location.resolve())
        commmands_to_run.append(
            (
                f"clang-tidy  -quiet {cc_file} > {log_file_location} 2>&1 ",
                log_file_location,
            )
        )

    # Total number of errors summed across processes
    total_errors = 0
    # Map from log_path: process
    current_processes = {}
    while len(commmands_to_run) > 0:
        for log_file_location in current_processes:
            process = current_processes[log_file_location]
            status = process.poll()
            if status == None:
                continue

            total_errors += status
            print(f"Found {status} errors for file with log path {log_file_location}.")
            if status > 0:
                with open(log_file_location) as f:
                    print(f.read())

            current_processes.pop(log_file_location)

        while (
            len(commmands_to_run) > 0 and len(current_processes.keys()) < max_concurrent
        ):
            new_command_to_run, new_log_file = commmands_to_run.pop()
            print(new_command_to_run)
            current_processes[new_log_file] = subprocess.Popen(
                new_command_to_run, shell=True
            )

        time.sleep(1000)

    if total_errors > 0:
        exit(total_errors)


def main():

    os.chdir(universe_dir())

    ensure_compile_commands_db_created()

    pragma_once_check()

    run_clang_tidy()


if __name__ == "__main__":
    # See https://stackoverflow.com/q/320232
    main()

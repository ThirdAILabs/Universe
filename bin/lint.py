#!/usr/bin/env python3

import multiprocessing
import os
import subprocess
import time
from pathlib import Path


# Checks whether a given filename is a dependency or in the build directory
# (and so we should not lint it)
def not_our_file(filename):
    return "/deps/" in filename or "/build/" in filename


# Returns all files that match the passed in glob pattern and that are one of
# "our" files (not a build artifact or a dependency)
def get_our_files(pattern):
    for filename in Path(".").rglob(pattern):
        filename = str(filename.resolve())
        if not_our_file(filename):
            continue
        yield filename


def pragma_once_check():
    files_that_need_pragma = []
    for header in get_our_files("*.h"):
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


# Check if any processes are finished, and if so collect their status. Return
# the sum of errors found across finished processes.
def reap_processes(log_path_to_running_process, log_paths):
    errors_found = 0
    for log_path in log_paths:
        process = log_path_to_running_process[log_path]
        status = process.poll()
        if status == None:
            continue

        errors_found += status
        print(f"Found {status} errors for file with log path {log_path}.")
        if status > 0:
            with open(log_path) as f:
                print(f.read())

        log_path_to_running_process.pop(log_path)
    return errors_found


# Starts new processes from commands_to_run (and pops them from the list) if
# there is enough room in the number of current processes running
def try_starting_new_processes(
    commands_to_run, log_path_to_running_process, max_concurrent
):
    while (
        len(commands_to_run) > 0 and len(log_path_to_running_process) < max_concurrent
    ):
        new_command_to_run, new_log_file = commands_to_run.pop()
        print(new_command_to_run)
        log_path_to_running_process[new_log_file] = subprocess.Popen(
            new_command_to_run, shell=True
        )


# Returns a list of all (clang tidy command, log_path) pairs for all .cc file
# we need to lint
def get_clang_tidy_commands_to_run():
    commands_to_run = []

    for cc_file in list(get_our_files("*.cc")):
        from pathlib import Path

        log_path = Path("clang_tidy_logs") / Path(cc_file).relative_to(
            universe_dir()
        ).with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = str(log_path.resolve())
        commands_to_run.append(
            (f"clang-tidy  -quiet {cc_file} > {log_path} 2>&1 ", log_path)
        )
    return commands_to_run


# This function runs clang-tidy on all .cc files in our codebase in parallel.
# It first gets a list of all .cc files and creates a list of clang-tidy command
# line invocations and associated log file. Then, it runs up to max_concurrent
# of the commands in parallel and monitors for when commands finish, replacing
# them with the next command from the list if there are any left. It keeps doing
# that until all commands have finished. If there were errors, this function
# exits() with the number of errors and prints the logs from each of the files
# with errors. Otherwise, it just returns.
def run_clang_tidy():
    max_concurrent_jobs = int(1.5 * multiprocessing.cpu_count())

    # Total number of errors summed across processes
    total_errors = 0

    # Map from log_path: process
    log_path_to_running_process = {}

    commands_to_run = get_clang_tidy_commands_to_run()

    while len(commands_to_run) > 0 or len(log_path_to_running_process) > 0:
        log_paths = list(log_path_to_running_process.keys())
        total_errors += reap_processes(log_path_to_running_process, log_paths)
        try_starting_new_processes(
            commands_to_run, log_path_to_running_process, max_concurrent_jobs
        )
        time.sleep(1)

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

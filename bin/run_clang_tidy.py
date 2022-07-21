#!/usr/bin/env python3

import argparse
import os
import subprocess
import glob


def get_changed_files():
    os.system("pwd")

    result = subprocess.run(
        ["git", "diff", "origin/main", "--name-only"], stdout=subprocess.PIPE
    )

    files = []
    for file in result.stdout.splitlines():
        filename = file.decode("utf-8")
        if filename.endswith(".cc") or filename.endswith(".h"):
            files.append(filename)
    return files


def get_all_cpp_files():
    files = []
    for file in glob.glob("**/*.cc", recursive=True):
        if not file.startswith("deps"):
            files.append(file)
    for file in glob.glob("**/*.h", recursive=True):
        if not file.startswith("deps"):
            files.append(file)
    return files


def main():
    bin_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(bin_directory + "/../")

    parser = argparse.ArgumentParser(
        description="Runs clang tidy on C++ code in Universe"
    )

    parser.add_argument(
        "--changed_files_only",
        action="store_true",
        help="Only run on the files that differ from main, i.e. changed on the given branch.",
    )

    args = parser.parse_args()

    if args.changed_files_only:
        files_to_lint = get_changed_files()
    else:
        files_to_lint = get_all_cpp_files()

    files_failed = []
    for file in files_to_lint:
        print(f"Running clang tidy on {file}...")
        exit_code = os.system(f"clang-tidy {file}")
        if exit_code != 0:
            files_failed.append(file)
        print("\n")

    if len(files_failed):
        print("The following files have lint errors:")
        for file in files_failed:
            print(f"\t'{file}'")
        exit(1)
    else:
        print("All files passed.")
        exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import os
import glob

def file_includes_header(header, filename):
    with open(filename) as file:
        for line in file.readlines():
            if header in line:
                return True
    return False


def get_all_cpp_files_including_header(header):
    files = []
    for file in glob.glob("**/*.cc", recursive=True):
        if not file.startswith("deps") and file_includes_header(header, file):
            files.append(file)
    for file in glob.glob("**/*.h", recursive=True):
        if not file.startswith("deps") and file_includes_header(header, file):
            files.append(file)
    return files


def main():
    bin_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(bin_directory + "/../")

    parser = argparse.ArgumentParser(
        description="Runs clang tidy on C++ code in Universe"
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Only run on the files that differ from main, i.e. changed on the given branch.",
    )


    args = parser.parse_args()
    print("Args.file = " + args.file)

    files_to_lint = []
    if args.file.endswith(".cc"):
        files_to_lint = [args.file]
    elif args.file.endswith(".h"):
        files_to_lint = get_all_cpp_files_including_header(args.file)
        files_to_lint.append(args.file)

    files_passed = []
    files_failed = []
    for file in files_to_lint:
        print(f"Running clang-tidy on {file}...")
        exit_code = os.system(f"clang-tidy {file}")
        if exit_code == 0:
            files_passed.append(file)
        else:
            files_failed.append(file)
        print("Done.\n")

    print("The following files passed clang-tidy:")
    for file in files_passed:
        print(f"\t'{file}'")
    print("\nThe following files failed clang-tidy:")
    for file in files_failed:
        print(f"\t'{file}'")


if __name__ == "__main__":
    main()

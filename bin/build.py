#!/usr/bin/env python3

import argparse
import multiprocessing
import os
import re


def parse_feature_flag(flag):

    flag = flag.upper()

    pattern = re.compile("^THIRDAI[A-Za-z_]*[A-Za-z]$")
    if not pattern.match(flag):
        raise argparse.ArgumentTypeError(
            f"""
            Feature flag {flag} doesn't follow {pattern.pattern}. Flag must 
            start with THIRDAI, then contain any combination of lower/upper 
            case and underscores, and then the last character cannot be an
            underscore.
            """
        )

    return flag


def main():

    bin_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(bin_directory)

    parser = argparse.ArgumentParser(description="Build a target in Universe")
    parser.add_argument(
        "-m",
        "--build_mode",
        default="Release",
        choices=["Release", "RelWithDebInfo", "RelWithAsan", "Debug", "DebugWithAsan"],
        metavar="MODE",  # Don't print the choices because they're ugly
        help='The releast mode to build with (see CMakeLists.txt for the specific compiler flags for each mode). Default is "Release".',
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default="-1",  # we check for -1 below, and if so set # jobs equal to 2 * total # threads
        type=int,
        help="Number of parallel jobs to run make with. Default is 2 * total # threads on the machine.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the commands that make is running.",
    )
    parser.add_argument(
        "-f",
        "--feature_flags",
        nargs="+",
        default=[],
        metavar="",  # Don't print the metavar because it's ugly
        type=parse_feature_flag,
        help="Whitespace seperated preprocessor flags to pass to the compiler to turn on and off features.",
    )
    args = parser.parse_args()

    # See https://stackoverflow.com/questions/414714/compiling-with-g-using-multiple-cores
    # for why we use 2 * num threads
    if args.jobs == -1:
        args.jobs = int(2 * multiprocessing.cpu_count())

    # Make sure build directory exists and cd to it
    os.system('mkdir -p "../build"')

    # Add THIRDAI_EXPOSE_ALL to the feature flag list, since this is an internal build
    if "THIRDAI_EXPOSE_ALL" not in args.feature_flags:
        args.feature_flags.append("THIRDAI_EXPOSE_ALL")

    # Create feature flag list for cmake
    # https://stackoverflow.com/questions/33242956/cmake-passing-lists-on-command-line
    joined_feature_flags = " ".join(args.feature_flags)

    # Change dir to top level, set environment variables, and run pip install
    os.environ["THIRDAI_BUILD_MODE"] = args.build_mode
    os.environ["THIRDAI_FEATURE_FLAGS"] = joined_feature_flags
    os.environ["THIRDAI_NUM_JOBS"] = str(args.jobs)

    os.chdir("..")
    os.system("pip3 install . --verbose --force")


if __name__ == "__main__":
    main()

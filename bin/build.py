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
            start with THIRDAI, can then contain any combination of lower/upper 
            case and underscores, and the last character cannot be an
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
        "-t",
        "--target",
        default="all",
        help='CMake target to build. Default is "all", i.e. build everything.',
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
    os.chdir("../build")

    # Create feature flag list for cmake
    # https://stackoverflow.com/questions/33242956/cmake-passing-lists-on-command-line
    joined_feature_flags = "\;".join(args.feature_flags)

    # Create cmake and make commands
    cmake_command = f"cmake .. -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE={args.build_mode} -DFEATURE_FLAGS={joined_feature_flags}"
    make_command = f"make {args.target} -s -j {args.jobs}"

    if args.verbose:
        make_command += " VERBOSE=1"

    # Run cmake and make commands
    os.system(cmake_command)
    os.system(make_command)


if __name__ == "__main__":
    main()

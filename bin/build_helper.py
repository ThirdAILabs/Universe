import argparse
import multiprocessing
import os


def parse_feature_flag(flag):

    # The flag needs to start with THIRDAI
    if not flag.startswith("THIRDAI"):
        raise argparse.ArgumentTypeError(
            f"Feature flag {flag} doesn't start with THIRDAI."
        )

    # Every char needs to be uppercase or an _
    for i, c in enumerate(flag):
        if c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_":
            raise argparse.ArgumentTypeError(
                f"In feature flag {flag}, char {c} at position {i} is not uppercase or an underscore."
            )

    # The first and last char should not be _
    if flag[0] == "_":
        raise argparse.ArgumentTypeError(
            f"Feature flag {flag} starts with an _ but should start with an uppercase letter."
        )
    if flag[-1] == "_":
        raise argparse.ArgumentTypeError(
            f"Feature flag {flag} ends with an _ but should end with an uppercase letter."
        )

    return flag


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
    default="-1",
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
    help="Whitespace seperated preprocessor flags to pass to the compiler to turn on and off features. These shou",
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

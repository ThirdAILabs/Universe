import argparse
import multiprocessing
import os

parser = argparse.ArgumentParser(description="Build a target in Universe")
parser.add_argument(
    "--build_mode",
    default="Release",
    choices=["Release", "RelWithDebInfo", "RelWithAsan", "Debug", "DebugWithAsan"],
    help='The releast mode to build with (see CMakeLists.txt for the specific compiler flags for each mode). Default is "Release".',
)
parser.add_argument(
    "--target",
    default="all",
    help='CMake target to build. Default is "all", i.e. build everything.',
)
parser.add_argument(
    "--jobs",
    default="-1",
    type=int,
    help="Number of parallel jobs to run make with. Default is int(1.5 * total # threads on the machine).",
)
parser.add_argument(
    "--flags",
    nargs="+",
    default=[],
    help="Whitespace seperated preprocessor flags to pass to the compiler to turn on and off features.",
)
args = parser.parse_args()

if (args.jobs == -1):
  args.jobs = int(1.5 * multiprocessing.cpu_count())

cmake_command = f"cmake .. -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE={args.build_mode}"
make_command = f"make {args.target} -s -j {args.jobs}"


os.system(' mkdir -p "../build"')
os.system('cd ../build')
os.system(cmake_command)
os.system(make_command)
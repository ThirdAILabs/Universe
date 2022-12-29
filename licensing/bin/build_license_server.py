#!/usr/bin/env python3

import argparse
import os
import pathlib

parser = argparse.ArgumentParser(description="Build the executable file for an on-prem go license server.")
parser.add_argument(
    "max_num_machines",
    type=int,
    help="Maximum number of machines permitted to be active at a single time. This will be built into the binary.",
)
parser.add_argument(
    "--machine_timeout_ms",
    type=int,
    help="Timeout in milliseconds on server beyond which a machine is not considered active anymore. This will be built into the binary. The default is currently 100 seconds (100000 ms).",
)
args = parser.parse_args()

compile_time_assignments = f"-X main.MaxActiveMachinesString={args.max_num_machines} "
if args.machine_timeout_ms != None:
    compile_time_assignments += f"-X main.ActiveTimeoutMillisString={args.machine_timeout_ms} "

licensing_bin_directory = parent_dir = pathlib.Path(__file__).parent
go_src_directory = licensing_bin_directory.parent / "src" / "server"
os.chdir(go_src_directory)

os.system(f"go build -ldflags \"{compile_time_assignments}\" -o license-server-max-{args.max_num_machines}")


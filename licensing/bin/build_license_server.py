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
args = parser.parse_args()

max_num_machines = args.max_num_machines

licensing_bin_directory = parent_dir = pathlib.Path(__file__).parent
go_src_directory = licensing_bin_directory.parent / "src" / "server"
os.chdir(go_src_directory)

os.system(f"go build -ldflags \"-X main.MaxActiveMachinesString={max_num_machines}\" -o license-server-max-{max_num_machines}")


#!/usr/bin/env python3

import os
import pathlib

licensing_bin_directory = parent_dir = pathlib.Path(__file__).parent
go_src_directory = licensing_bin_directory.parent / "src" / "server"
os.chdir(go_src_directory)
os.system(f"go test")

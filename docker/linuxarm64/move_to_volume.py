import shutil
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("src", type=Path)
parser.add_argument("dest", type=Path)
args = parser.parse_args()

# gather all files
allfiles = os.listdir(args.src)

# iterate on all files to move them to destination folder
for f in allfiles:
    src_path = os.path.join(args.src, f)
    dst_path = os.path.join(args.dest, f)
    shutil.move(src_path, dst_path)


import glob
import os
import sys
from typing import List


def get_all_cc_files() -> List[str]:
    cc_files = glob.glob("./**/*.cc", recursive=True)
    return [x for x in cc_files if not x.startswith("./deps") and not x.startswith("./build")]


def get_all_files_including_header(header: str) -> List[str]:
    cc_files = get_all_cc_files()

    includes = []
    for filename in cc_files:
        with open(filename) as file:
            for line in file:
                if header in line:
                    includes.append(filename)
                    break
    return includes


def invoke_clang_tidy(filename: str) -> bool:
    print(f"Running clang-tidy on file {filename}")
    exit_code = os.system(f"clang-tidy --quiet {filename}")
    return exit_code == 0


def main():
    if len(sys.argv) != 2:
        raise ValueError("Must pass in filename to run clang-tidy on.")

    root_directory= os.path.dirname(os.path.realpath(__file__)) + "/../"
    os.chdir(root_directory)

    
    if sys.argv[1].endswith(".h"):
        header = sys.argv[1].split("/")[-1]
        sources_to_lint = get_all_files_including_header(header)
        
        print(f"Linting {sys.argv[1]}. Included in:")
        for source in sources_to_lint:
            print(f"\t{source}")

        if not all([invoke_clang_tidy(source) for source in sources_to_lint]):
            exit(1)
    elif sys.argv[1].endswith(".cc"):
        if not invoke_clang_tidy(sys.argv[1]):
            exit(1)


if __name__ == "__main__":
    main()
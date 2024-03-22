import glob
import os
import shutil
import subprocess
import tempfile

DEMO_URL = "https://github.com/ThirdAILabs/Demos.git"

IGNORED_NBS = [
    "DeployThirdaiwithDatabricks.ipynb",
]


def skip_nb(path):
    return any([nb in path for nb in IGNORED_NBS])


def get_relative_notebook_paths(temp_dir):
    # Collect all of the jupyter notebooks in the Demos repo
    subprocess.call(["git", "clone", DEMO_URL], cwd=temp_dir)
    notebook_dir = os.path.join(temp_dir, "Demos", "**", "*.ipynb")
    notebook_paths = glob.glob(notebook_dir, recursive=True)
    notebook_paths = [path for path in notebook_paths if not skip_nb(path)]
    len_demos_dir = len(str(os.path.join(temp_dir, "Demos"))) + 1  # For slash
    relative_notebook_paths = [str(path)[len_demos_dir:] for path in notebook_paths]

    # Clean up. We don't need to keep this since we only need it to get the paths.
    shutil.rmtree(temp_dir)

    return relative_notebook_paths


def main():
    temp_dir = tempfile.mkdtemp()
    relative_notebook_paths = get_relative_notebook_paths(temp_dir)
    successes, failures = [], []
    for notebook_path in relative_notebook_paths:
        retcode = os.system(
            f'docker run -e "OPENAI_API_KEY=$OPENAI_API_KEY" -e "THIRDAI_KEY=$THIRDAI_KEY" thirdai/run_demos_build bash -c "python3 run_single_demo_notebook.py {notebook_path}"'
        )
        if retcode == 0:
            successes.append(notebook_path)
        else:
            failures.append(notebook_path)

    print("\nThe following notebooks have passed:\n" + "\n".join(successes))
    print("\nThe following notebooks have failed:\n" + "\n".join(failures))

    if failures:
        raise ValueError("The following notebooks failed: " + " ".join(failures))


if __name__ == "__main__":
    main()

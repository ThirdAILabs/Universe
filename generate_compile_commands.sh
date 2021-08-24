# Command to generate the compilation database file.
bazel build :create_comp_db

# Location of the compilation database file.
outfile="$(bazel info bazel-bin)/compile_commands.json"

# Command to replace the marker for exec_root in the file.
execroot=$(bazel info execution_root)
sed -i.bak "s@__EXEC_ROOT__@${execroot}@" "${outfile}"

rm -f compile_commands.json
cp ${outfile} compile_commands.json 
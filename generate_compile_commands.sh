# Command to generate the compilation database file.
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Location of the compilation database file.
outfile="build/compile_commands.json"

# TODO: do we still need this with cmake 
# Command to replace the marker for exec_root in the file.
#execroot=$(bazel info execution_root)
#sed -i.bak "s@__EXEC_ROOT__@${execroot}@" "${outfile}"
#sed -i '' -e 's/-fno-canonical-system-headers/ /g' "${outfile}"

rm -f compile_commands.json
cp ${outfile} compile_commands.json 
rm -rf build
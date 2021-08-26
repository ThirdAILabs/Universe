# Run clang-format on all files/directories recursively
find ../ -iname '*.h' -o -iname '*.cc' | xargs clang-format -i -style=file

# Check git diff for any changes
if git diff-index --quiet HEAD --
then
    # No changes
    :
else
    # Changes
    exit 1
fi
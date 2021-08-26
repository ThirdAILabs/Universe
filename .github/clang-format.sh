# Check git diff for any changes
if git diff-index --quiet HEAD --
then
    # No changes
    continue
else
    # Changes
    exit 1
fi
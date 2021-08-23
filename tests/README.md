Run the following for build results, from within the outermost directory (with WORKSPACE file).
```
bazel build //tests/:hash
bazel-bin/tests/hash
```

Run the following for MurmurHash and Tabulation Hash test results.
```
bazel test //tests/:hash_test
bazel-bin/tests/hash_test
```
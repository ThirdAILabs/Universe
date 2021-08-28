Run the following for build results, from within the outermost directory (with WORKSPACE file).
```
bazel build //examples/:hash
bazel-bin/examples/hash
```

Run the following for MurmurHash and Tabulation Hash test results.
```
bazel test //examples/:hash_test
bazel-bin/examples/hash_test
```
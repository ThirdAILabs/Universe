From within the outermost directory (with WORKSPACE file):

Run the following for build results on hash example.
```
bazel build //examples/:hash
bazel-bin/examples/hash
```

Run the following for MurmurHash and Tabulation Hash test results.
```
bazel test //examples/:hash_test
bazel-bin/examples/hash_test
```

Run the following for testing mpi-example.
```
bazel build --config=mpi_config //examples/mpi-example:mpi_hello_world
load("@com_grail_bazel_compdb//:aspects.bzl", "compilation_database")

compilation_database(
    name = "create_comp_db",
    targets = [
        "all_binary",
        "all_test"
    ],
    testonly = True
)

cc_binary(
  name = "all_binary",
  deps = [
    "//tests:hash",
  ],
)

cc_test(
  name = "all_test",
  deps = [
    "//tests:hash_test",
  ],
  testonly = True
)
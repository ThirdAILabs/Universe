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
    "//examples:hash",
    "//utils:utils"
  ],
)

cc_test(
  name = "all_test",
  deps = [
    "//examples:hash_test",
  ],
  testonly = True
)
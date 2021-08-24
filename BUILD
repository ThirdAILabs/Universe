load("@com_grail_bazel_compdb//:aspects.bzl", "compilation_database")

compilation_database(
    name = "create_comp_db",
    targets = [
        "//tests:hash",
        "//tests:hash_test"
    ],
    testonly = True
)

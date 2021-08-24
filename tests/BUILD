load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_binary(
    name = "hash",
    srcs = ["hash.cc"],
    deps = [
        "//tests/murmurhash:MurmurHash3",
        "//tests/tabulationhash:UniversalHash",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
  name = "hash_test",
  size = "small",
  srcs = ["hash_test.cc"],
  deps = [
        "@com_google_googletest//:gtest_main",
        "//tests/murmurhash:MurmurHash3",
        "//tests/tabulationhash:UniversalHash",
  ],
  visibility = ["//visibility:public"],
)
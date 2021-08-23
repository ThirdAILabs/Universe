load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "hello-greet",
    srcs = ["hello-greet.cc"],
    hdrs = ["hello-greet.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = [
        ":hello-greet",
        "//tutorial/lib:hello-time",
    ],
)

cc_test(
  name = "hello_world_test",
  size = "small",
  srcs = ["hello_world_test.cc"],
  deps = ["@com_google_googletest//:gtest_main"],
)

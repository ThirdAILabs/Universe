package(default_visibility = ["//visibility:public"])

filegroup(
	name = "empty",
	srcs = [],
)

filegroup(
	name = "toolchain",
	srcs = [
		":cc-compiler-amd64_openmpi",
	],
)

# Not sure why, but it prevents bazel from complaining "ERROR: The toolchain rule '//tools:cc-compiler-local' does not exist."
cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
)

cc_toolchain(
	name = "cc-compiler-amd64_openmpi",
	all_files = ":empty",
	compiler_files = ":empty",
	cpu = "amd64_openmpi",
	dwp_files = ":empty",
	dynamic_runtime_libs = [":empty"],
	linker_files = ":empty",
	objcopy_files = ":empty",
	static_runtime_libs = [":empty"],
	strip_files = ":empty",
	supports_param_files = 1,
)	
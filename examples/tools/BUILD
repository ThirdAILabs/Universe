package(default_visibility = ["//visibility:public"])
load(":cc_toolchain_config.bzl", "cc_toolchain_config")

filegroup(name = "empty")

cc_toolchain_suite(
	name = "mpi_toolchain",
	toolchains = {
		"openmpi": ":cc-compiler-amd64_openmpi",
        'k8': ":cc-compiler-amd64_openmpi",
	}
)

cc_toolchain(
    name = "cc-compiler-amd64_openmpi",
    toolchain_identifier = "cc-compiler-amd64_openmpi",
    toolchain_config = ":cc-compiler-amd64_openmpi_config",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
)

cc_toolchain_config(name = "cc-compiler-amd64_openmpi_config")
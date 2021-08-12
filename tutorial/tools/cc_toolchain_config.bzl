load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "tool_path")

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "/usr/bin/mpicc",
        ),
        tool_path(
            name = "ld",
            path = "/usr/bin/ld",
        ),
        tool_path(
            name = "ar",
            path = "/usr/bin/ar",
        ),
        tool_path(
            name = "cpp",
            path = "/bin/false",
        ),
        tool_path(
            name = "gcov",
            path = "/bin/false",
        ),
        tool_path(
            name = "nm",
            path = "/bin/false",
        ),
        tool_path(
            name = "objdump",
            path = "/bin/false",
        ),
        tool_path(
            name = "strip",
            path = "/bin/false",
        ),
    ]
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        cxx_builtin_include_directories = [
            "/usr/lib/llvm-9/lib/clang/10.0.0/include",
            "/usr/include/x86_64-linux-gnu/mpich",
            "/usr/include",
            "/usr/lib/x86_64-linux-gnu/openmpi/include",
            "/usr/lib/gcc/x86_64-linux-gnu/9/include",
        ],
        toolchain_identifier = "cc-compiler-amd64_openmpi",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "amd64_openmpi",
        target_libc = "unknown",
        compiler = "compiler",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
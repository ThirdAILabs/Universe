cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

# When I set LD_PRELOAD to be able use ASan with python tests, for some reason 
# CMake can't find Python unless we set it to an empty string.
set(ENV{LD_PRELOAD} "")

project(ThirdAI LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

message("====================================")
message("\tBUILD MODE: ${CMAKE_BUILD_TYPE}")
message("====================================")

add_compile_options(
  -fopenmp
  -fPIC
  -Wall
  -Wextra
  -Werror
  -Wno-unused-function
  -pedantic 
  
  $<$<CONFIG:Debug>:-Og>
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Debug>:-fno-omit-frame-pointer>

  $<$<CONFIG:DebugWithAsan>:-Og>
  $<$<CONFIG:DebugWithAsan>:-g>
  $<$<CONFIG:DebugWithAsan>:-fno-omit-frame-pointer>

  $<$<CONFIG:Release>:-DNDEBUG>
  $<$<CONFIG:Release>:-Ofast>
  $<$<CONFIG:Release>:-fno-finite-math-only>
  $<$<CONFIG:Release>:-funroll-loops>
  $<$<CONFIG:Release>:-ftree-vectorize>

  $<$<CONFIG:RelWithDebInfo>:-DNDEBUG>
  $<$<CONFIG:RelWithDebInfo>:-Ofast>
  $<$<CONFIG:RelWithDebInfo>:-fno-finite-math-only>
  $<$<CONFIG:RelWithDebInfo>:-funroll-loops>
  $<$<CONFIG:RelWithDebInfo>:-ftree-vectorize>
  $<$<CONFIG:RelWithDebInfo>:-g>
  $<$<CONFIG:RelWithDebInfo>:-fno-omit-frame-pointer>

  $<$<CONFIG:RelWithAsan>:-DNDEBUG>
  $<$<CONFIG:RelWithAsan>:-Ofast>
  $<$<CONFIG:RelWithAsan>:-fno-finite-math-only>
  $<$<CONFIG:RelWithAsan>:-funroll-loops>
  $<$<CONFIG:RelWithAsan>:-ftree-vectorize>
  $<$<CONFIG:RelWithAsan>:-g>
  $<$<CONFIG:RelWithAsan>:-fno-omit-frame-pointer>
)

add_link_options(
  -fopenmp
)

# Add feature flags passed in from python
add_compile_definitions(${FEATURE_FLAGS})

# In debug mode we are using ASan (address sanitizer) to provide better information on errors.
# We only run with this in debug mode because it carries a performace penalty. 
# See https://github.com/google/sanitizers/wiki/AddressSanitizer for more information.


# ASan does not work with GCC on M1 Macs yet. We would have to switch to clang. 
if(NOT ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64") 
  add_compile_options(
    $<$<CONFIG:RelWithAsan>:-fsanitize=address>
    $<$<CONFIG:DebugWithAsan>:-fsanitize=address>
  )

  add_link_options(
    $<$<CONFIG:RelWithAsan>:-fsanitize=address>
    $<$<CONFIG:DebugWithAsan>:-fsanitize=address>
  )
endif()

# It seems this gives us a small improvement on our blade server. Ideally we
# could run with march=native, which gives us performance improvements, but
# that would mean it wasn't portable to different cpus in a Docker container.
# TODO(any): Investigate whether we should do something similar for arm, and
# whether we can compile with march=native even with Docker.
if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  add_compile_options(
    $<$<CONFIG:Release>:-march=x86-64>
    $<$<CONFIG:RelWithDebInfo>:-march=x86-64>
    $<$<CONFIG:RelWithAsan>:-march=x86-64>
  )
endif()

find_package(OpenMP REQUIRED)

# Header only dependencies
include_directories(deps/pybind11/include)
include_directories(deps/cereal/include)
include_directories(deps/tomlplusplus/include/toml++)
include_directories(deps/eigen)

# Source dependencies
add_subdirectory(deps/pybind11)
add_subdirectory(deps/googletest)

# This is so we can include cryptopp/*.h. We want this even if the build flag
# isn't there so it always gets included in the compile_commands.json file and
# can work with vscode
include_directories(deps) 

# Check if there is a license flag, and only if so add the cryptopp lib
# We have two different License flags, one for if to build the license files
# (THIRDAI_BUILD_LICENSE) and one for if to check the license (THIRDAI_CHECK_LICENSE).
# If just the THIRDAI_CHECK_LICENSE flag is set there will be a linker error.
list (FIND FEATURE_FLAGS "THIRDAI_BUILD_LICENSE" _index)
SET (LICENSE_BUILD_FLAG_FOUND ${_index} GREATER -1)
if (${LICENSE_BUILD_FLAG_FOUND})

  # Install CRYPTOPP and set the static lib to be libcryptopp.a. This is pretty
  # ugly (the build happens at configure time, not build time) but honestly
  # okay (I spent way too long getting anything at all to work). Plus, it's 
  # gated by a feature flag anyways.
  # For now use default 32 jobs to build.
  set(CRYPTOPP_LIB ${CMAKE_SOURCE_DIR}/deps/cryptopp/libcryptopp.a)
  execute_process(COMMAND make static CXX=g++ -j32 -C ${CMAKE_SOURCE_DIR}/deps/cryptopp)

endif()


find_package (Python3 COMPONENTS Interpreter Development)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Enable testing
include(CTest)

# So we can include using e.g. #include <hashtable/src/SampledHashTable.h>
include_directories(".")

# Our source directories
add_subdirectory(bolt)
add_subdirectory(dataset)
add_subdirectory(exceptions)
add_subdirectory(flash)
add_subdirectory(hashing)
add_subdirectory(hashtable)
add_subdirectory(wrappers)

# Only add the licensing subdirectory if the corresponding flag is set
if (${LICENSE_BUILD_FLAG_FOUND})
  add_subdirectory(licensing)
endif()

file(
  GLOB BOLT_SOURCES 
  bolt/src/layers/*.cc 
  bolt/src/networks/*.cc 
  bolt/src/utils/*.cc 
)

file(
  GLOB_RECURSE FLASH_SOURCES 
  flash/src/*.cc 
)

file(
  GLOB_RECURSE HASHING_SOURCES
  hashing/src/*.cc
)

file(
  GLOB_RECURSE HASHTABLE_SOURCES
  hashtable/src/*.cc
)

file (
  GLOB_RECURSE WRAPPER_SOURCES
  wrappers/src/*.cc
)

file (
  GLOB_RECURSE DATASET_SOURCES
  dataset/src/*.cc
)

# pybind11_add_module automatically adds debug info to RelWithDebInfo and
# Debug builds, but not our ASan builds. This means that for now we can't run
# ASan from python, but honestly this is more trouble than it's worth so for
# now this is actually a feature rather than a bug.

pybind11_add_module(_thirdai 
  python_bindings/thirdai.cc 
  bolt/python_bindings/BoltPython.cc
  dataset/python_bindings/DatasetPython.cc
  flash/python_bindings/FlashPython.cc
  hashing/python_bindings/HashingPython.cc
  ${BOLT_SOURCES} 
  ${DLRM_SOURCES}
  ${FLASH_SOURCES} 
  ${HASHING_SOURCES}
  ${HASHTABLE_SOURCES}
  ${WRAPPER_SOURCES}
  ${DATASET_SOURCES}
)

if (${LICENSE_BUILD_FLAG_FOUND})
  target_link_libraries(_thirdai PRIVATE ${CRYPTOPP_LIB})
endif()

cmake_minimum_required(VERSION 3.4...3.18)

if(WIN32)
  # Set this to use static linking by default, see
  # https://cmake.org/cmake/help/latest/prop_tgt/MSVC_RUNTIME_LIBRARY.html
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# When I set LD_PRELOAD to be able use ASan with python tests, for some reason 
# CMake can't find Python unless we set it to an empty string.
set(ENV{LD_PRELOAD} "")

project(ThirdAI LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

message("====================================")
message("\tBUILD MODE: ${CMAKE_BUILD_TYPE}")
message("====================================")
message("\tBuilding with feature flags: ${FEATURE_FLAGS}")
separate_arguments(FEATURE_FLAGS)
message("\tFeature flags seperated into CMake List: ${FEATURE_FLAGS}")
message("====================================")


if (MSVC)
  if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    message( FATAL_ERROR "For now, we only support building in release mode using MSVC" )
  endif()
  set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS /GR /EHs /O2 /openmp:llvm /w")
else()
  add_compile_options(
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

endif()

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

find_package(OpenMP REQUIRED)

# Header only dependencies
include_directories(deps/pybind11/include)
include_directories(deps/cereal/include)
include_directories(deps/tomlplusplus/include/toml++)
include_directories(deps/eigen)
include_directories(deps/googletest/googletest/include)
include_directories(deps/googletest/googlemock/include)

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

  message("Building cryptopp...")
  # Install CRYPTOPP and set the static lib to be the static library file. This is pretty
  # ugly (the build happens at configure time, not build time) but honestly
  # okay (I spent way too long getting anything at all to work). Plus, it's 
  # gated by a feature flag anyways.
  # For now use default 32 jobs to build.
  if (MSVC)
    set(CRYPTOPP_LIB ${CMAKE_SOURCE_DIR}/deps/cryptopp/cryptlib.lib)
    message("${CRYPTOPP_LIB} must already exist or there will be errors (you can build with nmake, see https://www.cryptopp.com/wiki/Nmake_(Command_Line) )")
  else()
    set(CRYPTOPP_LIB ${CMAKE_SOURCE_DIR}/deps/cryptopp/libcryptopp.a)
    message("Running make in ${CMAKE_SOURCE_DIR}/deps/cryptopp")
    execute_process(COMMAND make static CXX=g++ -j32 -C ${CMAKE_SOURCE_DIR}/deps/cryptopp)
  endif()
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
add_subdirectory(search)
add_subdirectory(hashing)
add_subdirectory(hashtable)
add_subdirectory(wrappers)

# Only add the licensing subdirectory if the corresponding flag is set
if (${LICENSE_BUILD_FLAG_FOUND})
  add_subdirectory(licensing)
endif()

file(
  GLOB BOLT_SOURCES 
  bolt/src/auto_classifiers/*.cc
  bolt/src/graph/*.cc
  bolt/src/layers/*.cc 
  bolt/src/networks/*.cc 
  bolt/src/utils/*.cc 
)

file(
  GLOB_RECURSE SEARCH_SOURCES 
  search/src/*.cc 
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
  bolt/python_bindings/BoltGraphPython.cc
  dataset/python_bindings/DatasetPython.cc
  search/python_bindings/SearchPython.cc
  hashing/python_bindings/HashingPython.cc
  ${BOLT_SOURCES} 
  ${DLRM_SOURCES}
  ${SEARCH_SOURCES} 
  ${HASHING_SOURCES}
  ${HASHTABLE_SOURCES}
  ${DATASET_SOURCES}
  ${WRAPPER_SOURCES}
  ${DATASET_SOURCES}
)

target_link_libraries(_thirdai PRIVATE OpenMP::OpenMP_CXX)

if (${LICENSE_BUILD_FLAG_FOUND})
  target_link_libraries(_thirdai PRIVATE ${CRYPTOPP_LIB})
endif()

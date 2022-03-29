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
  
  $<$<CONFIG:Debug>:-O0>
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Debug>:-fno-omit-frame-pointer>

  $<$<CONFIG:Release>:-DNDEBUG>
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-funroll-loops>
  $<$<CONFIG:Release>:-ftree-vectorize>

  $<$<CONFIG:RelWithDebInfo>:-DNDEBUG>
  $<$<CONFIG:RelWithDebInfo>:-O3>
  $<$<CONFIG:RelWithDebInfo>:-funroll-loops>
  $<$<CONFIG:RelWithDebInfo>:-ftree-vectorize>
  $<$<CONFIG:RelWithDebInfo>:-g>
  $<$<CONFIG:RelWithDebInfo>:-fno-omit-frame-pointer>
)

add_link_options(
  -fopenmp
)

# In debug mode we are using ASan (address sanitizer) to provide better information on errors.
# We only run with this in debug mode because it carries a performace penalty. 
# See https://github.com/google/sanitizers/wiki/AddressSanitizer for more information.


# ASan does not work with GCC on M1 Macs yet. We would have to switch to clang. 
if(NOT ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64") 
  add_compile_options(
    $<$<CONFIG:DEBUG>:-fsanitize=address>
  )

  add_link_options(
    $<$<CONFIG:DEBUG>:-fsanitize=address>
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

pybind11_add_module(thirdai 
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
)

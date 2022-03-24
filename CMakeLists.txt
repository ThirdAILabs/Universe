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

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# If we need to add abseil for a warning, we can uncomment this:
# FetchContent_Declare(
#        abseil
#        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
#        GIT_TAG 20210324.2
# )
# FetchContent_MakeAvailable(abseil)

include(CTest)

# The following is for creating the python bindings

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.7.1
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

find_package (Python3 COMPONENTS Interpreter Development)

FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.0.1
)
FetchContent_MakeAvailable(tomlplusplus)
include_directories(${tomlplusplus_SOURCE_DIR})

set (JUST_INSTALL_CEREAL ON)
FetchContent_Declare(
    cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG        v1.3.2
)
FetchContent_MakeAvailable(
  cereal
)
include_directories(${cereal_SOURCE_DIR}/include)

# So we can include using e.g. #include <hashtable/src/SampledHashTable.h>
include_directories(".")
include_directories(${pybind11_SOURCE_DIR}/include)

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

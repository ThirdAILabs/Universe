cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp -fPIC -ffast-math -funroll-loops -ftree-vectorize -g")

project(ThirdAI LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(PythonLibs REQUIRED)

# One way that we can set global compiler flags for all targets. set(CMAKE_CXX_FLAGS) is the alternative.
# Add lots of warnings and all warnings as errors
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-unused-function)

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

find_package(OpenMP REQUIRED)

add_subdirectory(bolt)
add_subdirectory(utils)
add_subdirectory(flash)

file(
  GLOB BOLT_SOURCES 
  bolt/layers/FullyConnectedLayer.cc 
  bolt/networks/Network.cc 
  bolt/utils/*.cc 
  utils/hashing/DWTA.cc 
  utils/hashtable/SampledHashTable.cc
)

file(
  GLOB_RECURSE FLASH_SOURCES 
  flash/src/*.cc 
  utils/dataset/*.cc 
  utils/hashing/*.cc 
  utils/hashtable/*.cc
)

pybind11_add_module(thirdai python_bindings/thirdai.cc ${BOLT_SOURCES} ${FLASH_SOURCES})

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Ofast -fPIC -ffast-math -funroll-loops -ftree-vectorize")

project(ThirdAI LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(PythonLibs REQUIRED)

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

enable_testing()

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.2.3
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

add_subdirectory(examples)
add_subdirectory(bolt)
add_subdirectory(utils)

file(GLOB BOLT_SOURCES bolt/src/*.cc utils/hashing/DWTA.cc)
pybind11_add_module(thirdai python_bindings/thirdai.cc ${BOLT_SOURCES})

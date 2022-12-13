cmake_minimum_required(VERSION 3.4...3.18)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

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

# See [1] https://gcc.gnu.org/wiki/Visibility [2]
# https://stackoverflow.com/a/31157258/4565794
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Project versioning
find_package(Git REQUIRED)
include(GetVersionFromFile)
message(STATUS "Project name: ${PROJECT_NAME}")
message(STATUS "Project version: ${PROJECT_VERSION_STRING_FULL}")

message("====================================")
message("\tBUILD MODE: ${CMAKE_BUILD_TYPE}")
message("====================================")
message("\tBuilding with feature flags: ${THIRDAI_FEATURE_FLAGS}")
separate_arguments(THIRDAI_FEATURE_FLAGS)
message("\tFeature flags seperated into CMake List: ${THIRDAI_FEATURE_FLAGS}")
message("\tC++ compiler: ${CMAKE_CXX_COMPILER}")
message("\tC compiler: ${CMAKE_C_COMPILER}")
message("====================================")

option(THIRDAI_REPORT_BUILD_STEP_TIMES
       "Report time using builtin for make/ninja compile-build steps" OFF)

if(THIRDAI_REPORT_BUILD_STEP_TIMES)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
endif(THIRDAI_REPORT_BUILD_STEP_TIMES)

# Check if there is a license flag, and only if so add the cryptopp lib We have
# two different License flags, one for if to build the license files
# (THIRDAI_BUILD_LICENSE) and one for if to check the license
# (THIRDAI_CHECK_LICENSE). If just the THIRDAI_CHECK_LICENSE flag is set there
# will be a linker error.
list(FIND THIRDAI_FEATURE_FLAGS "THIRDAI_BUILD_LICENSE" _index)
set(LICENSE_BUILD_FLAG_FOUND ${_index} GREATER -1)
if(${LICENSE_BUILD_FLAG_FOUND})
  # Use our own source, also don't build tests.
  set(CRYPTOPP_BUILD_TESTING OFF)
  set(CRYPTOPP_SOURCES "${CMAKE_SOURCE_DIR}/deps/cryptopp")
  add_subdirectory(deps/cryptopp-cmake EXCLUDE_FROM_ALL)

  # For now only require openssl if we are building with licensing, since we
  # only communicate with the internet (keygen) in that case
  add_subdirectory(deps/cpp-httplib EXCLUDE_FROM_ALL)

  set(OPENSSL_USE_STATIC_LIBS TRUE)
  find_package(OpenSSL 1.1.1 REQUIRED)

  set(JSON_BuildTests
      OFF
      CACHE INTERNAL "")
  add_subdirectory(deps/json EXCLUDE_FROM_ALL)
endif()

if(MSVC)
  if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    message(
      FATAL_ERROR "For now, we only support building in release mode using MSVC"
    )
  endif()
  set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS /GR /EHs /O2 /w")

  # We create an interface to plant compile flags and be cohesive with the rest
  # of the OpenMP::OpenMP_CXX linkages. This converts project wide compile flags
  # narrowing it to just the required targets.

  add_library(_thirdai_windows_llvm_openmp INTERFACE)
  target_compile_options(_thirdai_windows_llvm_openmp INTERFACE "/openmp:llvm")
  add_library(OpenMP::OpenMP_CXX ALIAS _thirdai_windows_llvm_openmp)
else()

  find_package(OpenMP)
  if(NOT OpenMP_FOUND)

    # In this case, we were not able to find OpenMP. We are choosing to build
    # openmp from scratch here, using the LLVM standalone OpenMP source.
    #
    # In practice, this is intended for use for cross-compiling M1 (arm64)
    # wheels on MacOS GitHub runners (x86_64), where the possibility to use the
    # host's libomp is missing, and the library needs to be cross compiled for
    # target.

    # The following works with only newer CMake versions (Download), but we get
    # here only after all else has failed, and only for purposes of CI.

    # This block needs to happen before all the compile-options are added to
    # escape -Wall and -Werror being enforced on OpenMP library.

    set(LLVM_OPENMP_VERSION "13.0.1")
    set(LLVM_OPENMP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/openmp")

    if(NOT EXISTS ${LLVM_OPENMP_SOURCE_DIR})
      message(
        "Downloading openmp-${LLVM_OPENMP_VERSION} into ${LLVM_OPENMP_SOURCE_DIR}"
      )
      set(LLVM_OPENMP_DOWNLOAD_URL
          "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_OPENMP_VERSION}/openmp-${LLVM_OPENMP_VERSION}.src.tar.xz"
      )
      file(
        DOWNLOAD
        ${LLVM_OPENMP_DOWNLOAD_URL} # URL
        "${CMAKE_CURRENT_BINARY_DIR}/openmp-${LLVM_OPENMP_VERSION}.tar.bz2" # FILE
      )
      file(ARCHIVE_EXTRACT INPUT
           "${CMAKE_CURRENT_BINARY_DIR}/openmp-${LLVM_OPENMP_VERSION}.tar.bz2"
           DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
      file(RENAME
           "${CMAKE_CURRENT_BINARY_DIR}/openmp-${LLVM_OPENMP_VERSION}.src"
           "${LLVM_OPENMP_SOURCE_DIR}")
    endif()

    set(OPENMP_STANDALONE_BUILD ON)
    add_subdirectory(${LLVM_OPENMP_SOURCE_DIR} EXCLUDE_FROM_ALL)
    add_library(OpenMP::OpenMP_CXX ALIAS omp)

  endif(NOT OpenMP_FOUND)

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

endif()

set(THIRDAI_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Werror
    -Wno-unused-function
    -Wno-ignored-optimization-argument
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
    $<$<CONFIG:CompileAnalysis>:-DNDEBUG>
    $<$<CONFIG:CompileAnalysis>:-Ofast>
    $<$<CONFIG:CompileAnalysis>:-fno-finite-math-only>
    $<$<CONFIG:CompileAnalysis>:-funroll-loops>
    $<$<CONFIG:CompileAnalysis>:-ftree-vectorize>
    $<$<CONFIG:CompileAnalysis>:-ftime-trace>)

set(THIRDAI_ASAN_COMPILE_OPTIONS $<$<CONFIG:RelWithAsan>:-fsanitize=address>
                                 $<$<CONFIG:DebugWithAsan>:-fsanitize=address>)

# Custom command line options

# The following is added for Windows workflows failing on GitHub, inorder to
# provide a configurable TimeOut.  The default value is set to 5, from
# http://github.com/Kitware/CMake/blob/9d1ecd72fb45af722da7668d0c7482b7a0b1876f/Modules/GoogleTest.cmake#L419-L436,
# retaining default behaviour.
#
# For Windows workflow, we may configure this to a higher value now through
# CMake command-line supply.

set(THIRDAI_GTEST_DISCOVERY_TIMEOUT
    5
    CACHE STRING
          "Timeout for GoogleTest discovery, configurable from command-line.")

# Header only dependencies
include_directories(deps/eigen)

# Source dependencies
add_subdirectory(deps/pybind11 EXCLUDE_FROM_ALL)
add_subdirectory(deps/googletest EXCLUDE_FROM_ALL)
add_subdirectory(deps/spdlog EXCLUDE_FROM_ALL)
# Turn off prometheus-cpp's testing build so that we don't try to build two
# copies of gmock
option(ENABLE_TESTING "" OFF)
option(ENABLE_PUSH "" OFF)
add_subdirectory(deps/prometheus-cpp EXCLUDE_FROM_ALL)

# PERFORMANCE_COMPARISON is always enabled by default in cereal CMakeLists which
# requires the boost and we dont want to add the boost dependencies, hence
# skipping the performance comparison. Earlier we use to include only header(so
# not using any CMakeLists) but now we have used target based hence including
# CMakelists thats why we need this.
set(SKIP_PERFORMANCE_COMPARISON ON)
add_subdirectory(deps/cereal EXCLUDE_FROM_ALL)

# This is so we can include cryptopp/*.h. We want this even if the build flag
# isn't there so it always gets included in the compile_commands.json file and
# can work with vscode
include_directories(deps)

find_package(Python3 COMPONENTS Interpreter Development)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)

# Enable testing
include(CTest)

# So we can include using e.g. #include <hashtable/src/SampledHashTable.h>
include_directories(".")
include_directories(${CMAKE_BINARY_DIR})

# Our source directories
add_subdirectory(auto_ml)
add_subdirectory(bolt)
add_subdirectory(bolt_vector)
add_subdirectory(dataset)
add_subdirectory(search)
add_subdirectory(hashing)
add_subdirectory(hashtable)
add_subdirectory(compression)
add_subdirectory(utils)

set(THIRDAI_SOURCES
    bolt/src/graph/Graph.cc
    bolt/src/graph/DatasetContext.cc
    bolt/src/graph/Node.cc
    bolt/src/graph/nodes/Switch.cc
    bolt/src/graph/nodes/Concatenate.cc
    bolt/src/graph/nodes/DlrmAttention.cc
    bolt/src/graph/nodes/DotProduct.cc
    bolt/src/graph/nodes/Embedding.cc
    bolt/src/graph/nodes/FullyConnected.cc
    bolt/src/graph/nodes/LayerNorm.cc
    bolt/src/graph/nodes/Input.cc
    bolt/src/graph/InferenceOutputTracker.cc
    bolt/src/layers/EmbeddingLayer.cc
    bolt/src/layers/FullyConnectedLayer.cc
    bolt/src/layers/ConvLayer.cc
    bolt/src/layers/LayerConfig.cc
    bolt/src/layers/SamplingConfig.cc
    bolt/src/loss_functions/LossFunctions.cc
    bolt/src/metrics/Metric.cc
    search/src/MaxFlash.cc
    search/src/MaxFlashArray.cc
    search/src/Flash.cc
    hashing/src/DensifiedMinHash.cc
    hashing/src/DWTA.cc
    hashing/src/FastSRP.cc
    hashing/src/MurmurHash.cc
    hashing/src/MinHash.cc
    hashing/src/SRP.cc
    hashing/src/UniversalHash.cc
    hashtable/src/SampledHashTable.cc
    hashtable/src/VectorHashTable.cc
    licensing/src/CheckLicense.cc
    dataset/src/Vocabulary.cc
    telemetry/src/PrometheusClient.cc
    new_dataset/src/featurization_pipeline/transformations/Binning.cc
    new_dataset/src/featurization_pipeline/transformations/StringHash.cc
    new_dataset/src/featurization_pipeline/augmentations/ColdStartText.cc
    new_dataset/src/featurization_pipeline/ColumnMap.cc
    compression/src/CountSketch.cc
    compression/src/DragonVector.cc
    auto_ml/src/dataset_factories/SingleBlockDatasetFactory.cc
    auto_ml/src/dataset_factories/udt/DataTypes.cc
    auto_ml/src/dataset_factories/udt/FeatureComposer.cc
    auto_ml/src/dataset_factories/udt/UDTDatasetFactory.cc
    auto_ml/src/deployment_config/BlockConfig.cc
    auto_ml/src/deployment_config/DeploymentConfig.cc
    auto_ml/src/deployment_config/HyperParameter.cc
    auto_ml/src/deployment_config/ModelConfig.cc
    auto_ml/src/deployment_config/NodeConfig.cc
    auto_ml/src/deployment_config/dataset_configs/SingleBlockDatasetFactoryConfig.cc
    auto_ml/src/deployment_config/dataset_configs/UDTDatasetFactoryConfig.cc
    utils/Logging.cc)

# Only add the licensing subdirectory and licensing cc files if the
# corresponding flag is set
if(${LICENSE_BUILD_FLAG_FOUND})
  add_subdirectory(licensing)
  set(THIRDAI_SOURCES ${THIRDAI_SOURCES}
                      licensing/src/keygen/KeygenCommunication.cc)
endif()

add_library(thirdai STATIC ${THIRDAI_SOURCES})

target_link_libraries(thirdai PUBLIC OpenMP::OpenMP_CXX spdlog::spdlog
                                     cereal::cereal prometheus-cpp::pull)
if(${LICENSE_BUILD_FLAG_FOUND})
  target_link_libraries(
    thirdai PUBLIC cryptopp::cryptopp OpenSSL::Crypto OpenSSL::SSL
                   nlohmann_json::nlohmann_json)
endif()

# pybind11_add_module automatically adds debug info to RelWithDebInfo and Debug
# builds, but not our ASan builds. This means that for now we can't run ASan
# from python, but honestly this is more trouble than it's worth so for now this
# is actually a feature rather than a bug.

pybind11_add_module(
  _thirdai
  python_bindings/thirdai.cc
  bolt/python_bindings/BoltPython.cc
  bolt/python_bindings/BoltNNPython.cc
  bolt/python_bindings/CallbacksPython.cc
  dataset/python_bindings/DatasetPython.cc
  new_dataset/python_bindings/DatasetPython.cc
  new_dataset/python_bindings/FeaturizationPython.cc
  search/python_bindings/SearchPython.cc
  hashing/python_bindings/HashingPython.cc
  auto_ml/python_bindings/AutomlPython.cc
  auto_ml/python_bindings/DeploymentPython.cc
  auto_ml/src/models/ModelPipeline.cc
  auto_ml/src/models/OutputProcessor.cc
  auto_ml/src/models/UniversalDeepTransformer.cc
  telemetry/python_bindings/TelemetryPython.cc)

target_link_libraries(_thirdai PUBLIC thirdai)

if(NOT MSVC)
  target_compile_options(thirdai PRIVATE ${THIRDAI_COMPILE_OPTIONS})
  target_compile_options(_thirdai PRIVATE ${THIRDAI_COMPILE_OPTIONS})
endif()

# Add feature flags passed in from python
target_compile_definitions(thirdai PRIVATE ${THIRDAI_FEATURE_FLAGS})
target_compile_definitions(_thirdai PRIVATE ${THIRDAI_FEATURE_FLAGS})

# In debug mode we are using ASan (address sanitizer) to provide better
# information on errors. We only run with this in debug mode because it carries
# a performace penalty. See
# https://github.com/google/sanitizers/wiki/AddressSanitizer for more
# information.

target_compile_options(thirdai PUBLIC ${THIRDAI_ASAN_COMPILE_OPTIONS})

target_link_options(thirdai PUBLIC ${THIRDAI_ASAN_COMPILE_OPTIONS})

target_compile_options(_thirdai PRIVATE ${THIRDAI_ASAN_COMPILE_OPTIONS})

target_link_options(_thirdai PRIVATE ${THIRDAI_ASAN_COMPILE_OPTIONS})

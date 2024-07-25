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
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
option(THIRDAI_BUILD_OPENSSL_FROM_SOURCE "Build OpenSSL from source" OFF)

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
list(FIND THIRDAI_FEATURE_FLAGS "THIRDAI_CHECK_LICENSE" _index)
set(LICENSE_CHECK_FLAG_FOUND ${_index} GREATER -1)

if(${LICENSE_BUILD_FLAG_FOUND})
  # Use our own source, also don't build tests.
  set(CRYPTOPP_BUILD_TESTING OFF)
  set(CRYPTOPP_SOURCES "${CMAKE_SOURCE_DIR}/deps/cryptopp")
  add_subdirectory(deps/cryptopp-cmake EXCLUDE_FROM_ALL)

  # For now only require openssl if we are building with licensing, since we
  # only communicate with the internet (keygen) in that case
  add_subdirectory(deps/cpp-httplib EXCLUDE_FROM_ALL)
  find_package(OpenSSL 1.1.1 REQUIRED)

endif()

set(JSON_BuildTests
    OFF
    CACHE INTERNAL "")
add_subdirectory(deps/json EXCLUDE_FROM_ALL)

if(MSVC)
  if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    message(
      FATAL_ERROR "For now, we only support building in release mode using MSVC"
    )
  endif()
  # Read this for /permissive- https://stackoverflow.com/a/24414279/4565794
  set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS /GR /EHs /O2 /w /permissive-")

  # We create an interface to plant compile flags and be cohesive with the rest
  # of the OpenMP::OpenMP_CXX linkages. This converts project wide compile flags
  # narrowing it to just the required targets.

  add_library(_thirdai_windows_llvm_openmp INTERFACE)
  target_compile_options(_thirdai_windows_llvm_openmp INTERFACE "/openmp:llvm")
  add_library(OpenMP::OpenMP_CXX ALIAS _thirdai_windows_llvm_openmp)
else()

  find_package(OpenMP REQUIRED)

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

endif()

set(THIRDAI_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Werror
    -Wno-unused-function
    -Wno-ignored-optimization-argument
    -Wno-psabi
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
add_subdirectory(deps/utf8proc EXCLUDE_FROM_ALL)

# Turn off prometheus-cpp's testing build so that we don't try to build two
# copies of gmock
set(ENABLE_TESTING OFF)
set(ENABLE_PUSH OFF)
# add_subdirectory(deps/prometheus-cpp EXCLUDE_FROM_ALL)

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

# Note: for local builds need to use clang++ not g++ because of pointer
# conversion error in rocksdb

if(NOT WIN32)
  # These options are prevent rocksdb from linking a second version gtest which
  # causes a conflict with our gtest library. This does not impact our use for
  # gtest.
  option(WITH_TESTS OFF)
  option(WITH_BENCHMARK_TOOLS OFF)
  option(WITH_TOOLS OFF)

  # This is just so that we don't need to complile and link with this library.
  option(WITH_GFLAGS OFF)

  option(ROCKSDB_BUILD_SHARED OFF)

  # This is so we can implement a MergeOperator:
  # https://github.com/facebook/rocksdb/issues/3811
  set(USE_RTTI
      "TRUE"
      CACHE STRING "")
  add_subdirectory(deps/rocksdb)
  include_directories(deps/rocksdb/include)
endif()

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
add_subdirectory(data)
add_subdirectory(dataset)
add_subdirectory(search)
add_subdirectory(hashing)
add_subdirectory(hashtable)
add_subdirectory(mach)
add_subdirectory(archive)
add_subdirectory(compression)
add_subdirectory(utils)

if(${LICENSE_BUILD_FLAG_FOUND})
  add_subdirectory(licensing)
endif()

set(THIRDAI_SOURCES
    bolt_vector/src/BoltVector.cc
    bolt/src/nn/autograd/Computation.cc
    bolt/src/nn/autograd/ComputationGraph.cc
    bolt/src/nn/model/AllocationManager.cc
    bolt/src/nn/model/Model.cc
    bolt/src/nn/ops/Op.cc
    bolt/src/nn/ops/FullyConnected.cc
    bolt/src/nn/ops/Embedding.cc
    bolt/src/nn/ops/RobeZ.cc
    bolt/src/nn/ops/Concatenate.cc
    bolt/src/nn/ops/Input.cc
    bolt/src/nn/ops/LayerNorm.cc
    bolt/src/nn/ops/Switch.cc
    bolt/src/nn/ops/Activation.cc
    bolt/src/nn/ops/DlrmAttention.cc
    bolt/src/nn/ops/DotProduct.cc
    bolt/src/nn/ops/CosineSimilarity.cc
    bolt/src/nn/ops/PatchEmbedding.cc
    bolt/src/nn/ops/PatchSum.cc
    bolt/src/nn/ops/WeightedSum.cc
    bolt/src/nn/ops/MaxPool1D.cc
    bolt/src/nn/ops/QuantileMixing.cc
    bolt/src/nn/tensor/Tensor.cc
    bolt/src/nn/loss/Loss.cc
    bolt/src/nn/loss/ComparativeLoss.cc
    bolt/src/nn/loss/CategoricalCrossEntropy.cc
    bolt/src/nn/loss/BinaryCrossEntropy.cc
    bolt/src/nn/loss/EuclideanContrastive.cc
    bolt/src/nn/loss/ExternalLoss.cc
    bolt/src/nn/optimizers/Optimizer.cc
    bolt/src/nn/optimizers/Adam.cc
    bolt/src/nn/optimizers/SGD.cc
    bolt/src/neuron_index/NeuronIndex.cc
    bolt/src/neuron_index/LshIndex.cc
    bolt/src/neuron_index/RandomSampler.cc
    bolt/src/neuron_index/MachNeuronIndex.cc
    bolt/src/train/metrics/Metric.cc
    bolt/src/train/callbacks/Callback.cc
    bolt/src/train/metrics/LossMetric.cc
    bolt/src/train/metrics/CategoricalAccuracy.cc
    bolt/src/train/metrics/PrecisionAtK.cc
    bolt/src/train/metrics/RecallAtK.cc
    bolt/src/train/metrics/FMeasure.cc
    bolt/src/train/metrics/MachPrecision.cc
    bolt/src/train/metrics/MachRecall.cc
    bolt/src/train/trainer/Trainer.cc
    bolt/src/train/trainer/Dataset.cc
    bolt/src/layers/EmbeddingLayer.cc
    bolt/src/layers/FullyConnectedLayer.cc
    bolt/src/layers/LayerConfig.cc
    bolt/src/layers/SamplingConfig.cc
    bolt/src/inference/EmbFcInference.cc
    bolt/src/NER/model/NerUDTModel.cc
    bolt/src/NER/model/NerBoltModel.cc
    bolt/src/NER/model/NerClassifier.cc
    bolt/src/NER/model/NER.cc
    bolt/src/text_generation/ContextualModel.cc
    bolt/src/text_generation/DyadicModel.cc
    bolt/src/text_generation/GenerativeModel.cc
    bolt/src/root_cause_analysis/RCA.cc
    bolt/src/metrics/Metric.cc
    bolt/src/seismic/SeismicLabels.cc
    search/src/MaxFlash.cc
    search/src/MaxFlashArray.cc
    search/src/Flash.cc
    search/src/inverted_index/InvertedIndex.cc
    search/src/inverted_index/ShardedRetriever.cc
    search/src/inverted_index/IndexConfig.cc
    search/src/inverted_index/FinetunableRetriever.cc
    search/src/inverted_index/Tokenizer.cc
    hashing/src/DensifiedMinHash.cc
    hashing/src/DWTA.cc
    hashing/src/FastSRP.cc
    hashing/src/MurmurHash.cc
    hashing/src/MinHash.cc
    hashing/src/SRP.cc
    hashing/src/UniversalHash.cc
    hashtable/src/SampledHashTable.cc
    hashtable/src/VectorHashTable.cc
    dataset/src/DataSource.cc
    dataset/src/InMemoryDataset.cc
    dataset/src/VectorBuffer.cc
    dataset/src/dataset_loaders/DatasetLoader.cc
    dataset/src/cold_start/ColdStartDataSource.cc
    dataset/src/utils/TokenEncoding.cc
    dataset/src/featurizers/llm/TextContextFeaturizer.cc
    dataset/src/featurizers/llm/TextGenerationFeaturizer.cc
    dataset/src/featurizers/llm/TextClassificationFeaturizer.cc
    dataset/src/utils/CsvParser.cc
    dataset/src/blocks/ColumnIdentifier.cc
    dataset/src/featurizers/ClickThroughFeaturizer.cc
    dataset/src/featurizers/ProcessorUtils.cc
    dataset/src/featurizers/SvmFeaturizer.cc
    dataset/src/featurizers/TabularFeaturizer.cc
    dataset/src/blocks/GraphBlocks.cc
    dataset/src/blocks/BlockList.cc
    dataset/src/blocks/ColumnIdentifier.cc
    dataset/src/blocks/RecurrenceAugmentation.cc
    dataset/src/blocks/Sequence.cc
    dataset/src/blocks/text/Text.cc
    dataset/src/blocks/TabularHashFeatures.cc
    dataset/src/blocks/text/WordpieceTokenizer.cc
    dataset/src/blocks/text/TextTokenizer.cc
    dataset/src/mach/MachIndex.cc
    dataset/src/mach/MachBlock.cc
    dataset/src/ranking/KeywordOverlapRanker.cc
    dataset/src/utils/GraphInfo.cc
    # telemetry/src/PrometheusClient.cc
    data/src/transformations/Transformation.cc
    data/src/transformations/State.cc
    data/src/transformations/Binning.cc
    data/src/transformations/StringHash.cc
    data/src/transformations/TextTokenizer.cc
    data/src/transformations/TextCompat.cc
    data/src/transformations/cold_start/TextAugmentationUtils.cc
    data/src/transformations/cold_start/ColdStartText.cc
    data/src/transformations/cold_start/VariableLengthColdStart.cc
    data/src/transformations/FeatureHash.cc
    data/src/transformations/Tabular.cc
    data/src/transformations/Pipeline.cc
    data/src/transformations/MachLabel.cc
    data/src/transformations/MachMemory.cc
    data/src/transformations/AddMachMemorySamples.cc
    data/src/transformations/StringConcat.cc
    data/src/transformations/StringCast.cc
    data/src/transformations/StringIDLookup.cc
    data/src/transformations/CategoricalTemporal.cc
    data/src/transformations/NumericalTemporal.cc
    data/src/transformations/Tabular.cc
    data/src/transformations/Date.cc
    data/src/transformations/CrossColumnPairgrams.cc
    data/src/transformations/DeduplicateTokens.cc
    data/src/transformations/EncodePosition.cc
    data/src/transformations/Recurrence.cc
    data/src/transformations/CountTokens.cc
    data/src/transformations/Graph.cc
    data/src/transformations/RegressionBinning.cc
    data/src/transformations/DyadicInterval.cc
    data/src/transformations/NextWordPrediction.cc
    data/src/transformations/SpladeAugmentation.cc
    data/src/transformations/ner/NerTokenFromStringArray.cc
    data/src/transformations/ner/NerTokenizationUnigram.cc
    data/src/transformations/ner/NerDyadicDataProcessor.cc
    data/src/transformations/ner/rules/Rule.cc
    data/src/transformations/ner/rules/Pattern.cc
    data/src/transformations/ner/rules/CommonPatterns.cc
    data/src/transformations/ner/rules/IBANPattern.cc
    data/src/ColumnMap.cc
    data/src/columns/ArrayColumns.cc
    data/src/columns/ValueColumns.cc
    data/src/TensorConversion.cc
    data/src/Loader.cc
    data/src/ColumnMapIterator.cc
    data/src/rca/ExplanationMap.cc
    mach/src/MachRetriever.cc
    mach/src/MachConfig.cc
    mach/src/EnsembleSearch.cc
    archive/src/Archive.cc
    archive/src/Map.cc
    archive/src/List.cc
    archive/src/Value.cc
    archive/src/ParameterReference.cc
    compression/src/CountSketch.cc
    compression/src/DragonVector.cc
    auto_ml/src/featurization/DataTypes.cc
    auto_ml/src/featurization/TabularTransformations.cc
    auto_ml/src/featurization/Featurizer.cc
    auto_ml/src/featurization/MachFeaturizer.cc
    auto_ml/src/featurization/GraphFeaturizer.cc
    auto_ml/src/featurization/RecurrentFeaturizer.cc
    auto_ml/src/featurization/ReservedColumns.cc
    auto_ml/src/featurization/TabularBlockComposer.cc # Deprecate
    auto_ml/src/featurization/TabularDatasetFactory.cc # Deprecate
    auto_ml/src/cold_start/ColdStartUtils.cc # Deprecate
    auto_ml/src/pretrained/PretrainedBase.cc
    auto_ml/src/config/ModelConfig.cc
    auto_ml/src/config/FlashConfig.cc
    auto_ml/src/config/Parameter.cc
    auto_ml/src/rlhf/BalancingSamples.cc
    auto_ml/src/rlhf/RLHFSampler.cc # Deprecate
    auto_ml/src/cpp_classifier/CppClassifier.cc
    utils/src/SymSpellBackend/symspell.cc
    utils/src/SymSpellCpp/SymSpell.cc
    utils/text/PorterStemmer.cc
    utils/Logging.cc
    utils/text/StringManipulation.cc
    utils/UUID.cc
    utils/Random.cc
    licensing/src/entitlements/RestrictionTree.cc
    licensing/src/entitlements/Entitlements.cc
    versioning/src/Versions.cc)

if(NOT WIN32)
  # The OnDiskIndex doesn't work on windows because the bitfields for the
  # DocCount struct still result in a size of > 8 bytes, so it will not be
  # compatible with indexes created on mac/linux.
  set(THIRDAI_SOURCES
      ${THIRDAI_SOURCES} search/src/inverted_index/OnDiskIndex.cc
      search/src/inverted_index/adapters/RocksDbAdapter.cc
      search/src/inverted_index/adapters/RocksDbReadOnlyAdapter.cc)
endif()

# Only add the licensing subdirectory and licensing cc files if the
# corresponding flag is set
if(${LICENSE_CHECK_FLAG_FOUND})
  set(THIRDAI_SOURCES
      ${THIRDAI_SOURCES}
      # licensing/src/methods/keygen/KeygenCommunication.cc
      # licensing/src/methods/heartbeat/Heartbeat.cc
      licensing/src/CheckLicenseEnabled.cc
      # licensing/src/methods/keygen/KeyMethod.cc
      # licensing/src/methods/heartbeat/LocalServerMethod.cc
      licensing/src/methods/file/FileMethod.cc)
else()
  set(THIRDAI_SOURCES ${THIRDAI_SOURCES} licensing/src/CheckLicenseDisabled.cc)
endif()

add_library(thirdai STATIC ${THIRDAI_SOURCES})

# target_link_libraries(
#   thirdai PUBLIC OpenMP::OpenMP_CXX spdlog::spdlog cereal::cereal
#                  prometheus-cpp::pull utf8proc)
target_link_libraries(
  thirdai PUBLIC OpenMP::OpenMP_CXX spdlog::spdlog cereal::cereal utf8proc)
if(NOT WIN32)
  target_link_libraries(thirdai PUBLIC rocksdb)
endif()

if(${LICENSE_BUILD_FLAG_FOUND})
  # target_link_libraries(thirdai PUBLIC cryptopp::cryptopp OpenSSL::Crypto
  #                                      OpenSSL::SSL)
  target_link_libraries(thirdai PUBLIC cryptopp::cryptopp)
endif()

target_link_libraries(thirdai PUBLIC nlohmann_json::nlohmann_json)

add_library(thirdai_core SHARED ${THIRDAI_SOURCES})
# target_link_libraries(
#   thirdai_core
#   PUBLIC OpenMP::OpenMP_CXX spdlog::spdlog cereal::cereal prometheus-cpp::pull
#          utf8proc nlohmann_json::nlohmann_json)
         target_link_libraries(
  thirdai_core
  PUBLIC OpenMP::OpenMP_CXX spdlog::spdlog cereal::cereal
         utf8proc nlohmann_json::nlohmann_json)

if(${LICENSE_BUILD_FLAG_FOUND})
  # target_link_libraries(thirdai_core PUBLIC cryptopp::cryptopp OpenSSL::Crypto
  #                                           OpenSSL::SSL)
target_link_libraries(thirdai_core PUBLIC cryptopp::cryptopp)
endif()

# pybind11_add_module automatically adds debug info to RelWithDebInfo and Debug
# builds, but not our ASan builds. This means that for now we can't run ASan
# from python, but honestly this is more trouble than it's worth so for now this
# is actually a feature rather than a bug.

pybind11_add_module(
  _thirdai
  python_bindings/thirdai.cc
  bolt/python_bindings/PybindUtils.cc
  bolt/python_bindings/BoltNNPython.cc
  bolt/python_bindings/BoltTrainPython.cc
  bolt/python_bindings/BoltCompression.cc
  bolt/python_bindings/NumpyConversions.cc
  bolt/python_bindings/BoltTextGeneration.cc
  bolt/python_bindings/BoltNERPython.cc
  bolt/python_bindings/Porting.cc
  bolt/src/seismic/SeismicBase.cc
  bolt/src/seismic/SeismicEmbedding.cc
  bolt/src/seismic/SeismicClassifier.cc
  bolt/python_bindings/BoltSeismic.cc
  dataset/python_bindings/DatasetPython.cc
  data/python_bindings/DataPython.cc
  search/python_bindings/SearchPython.cc
  search/python_bindings/BeamSearch.cc
  hashing/python_bindings/HashingPython.cc
  auto_ml/python_bindings/AutomlPython.cc
  auto_ml/python_bindings/PretrainedBasePython.cc
  auto_ml/src/udt/UDT.cc
  auto_ml/src/udt/utils/Models.cc
  auto_ml/src/udt/utils/Classifier.cc
  auto_ml/src/udt/backends/UDTClassifier.cc
  auto_ml/src/udt/backends/UDTMach.cc
  auto_ml/src/udt/backends/DeprecatedUDTMachClassifier.cc # Deprecate
  auto_ml/src/udt/backends/UDTRecurrentClassifier.cc
  auto_ml/src/udt/backends/UDTRegression.cc
  auto_ml/src/udt/backends/UDTQueryReformulation.cc
  auto_ml/src/udt/backends/UDTGraphClassifier.cc
  auto_ml/src/udt/backends/UDTNer.cc
  auto_ml/src/udt/utils/Classifier.cc
  auto_ml/src/udt/utils/KwargUtils.cc
  mach/python_bindings/MachPython.cc
  telemetry/python_bindings/TelemetryPython.cc
  licensing/python_bindings/LicensingPython.cc)

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

#pragma once

#include <cstdint>

namespace thirdai::automl::udt::defaults {

// Dimension of hidden layer in default udt model.
constexpr uint32_t HIDDEN_DIM = 512;

// Default batch size used for training when not provided. Also used for
// inference and validation.
constexpr uint32_t BATCH_SIZE = 2048;

// Whether to freeze hash tables after first epoch of training.
constexpr bool FREEZE_HASH_TABLES = true;

// Whether to use sigmoid and bce for the output layer in UDT
constexpr bool USE_SIGMOID_BCE = false;

// Whether to use tanh for the hidden layers in UDT
constexpr bool USE_TANH = false;

// Whether to use tabular pairgrams.
constexpr bool CONTEXTUAL_COLUMNS = false;

// Time granularity for temporal tracking.
constexpr const char* TIME_GRANULARITY = "d";

// Only use pairgrams if phrase has up to this number of words.
constexpr uint32_t PAIRGRAM_WORD_LIMIT = 15;

// Used in temporal tracking.
constexpr uint32_t LOOKAHEAD = 0;

// Dimension to feature hash input to.
constexpr uint32_t FEATURE_HASH_RANGE = 100000;

// For parsing csv data.
constexpr char CSV_DELIMITER = ',';

// Maximum number of samples to use to tune binary classification threshold.
constexpr uint32_t MAX_SAMPLES_FOR_THRESHOLD_TUNING = 1000000;

// Number of thresholds to consider for binary classification threshold.
constexpr uint32_t NUM_THRESHOLDS_TO_CHECK = 1000;

// Number of bins to use for regression as classification.
constexpr uint32_t REGRESSION_BINS = 100;

// Radius around the correct bin that will also be counted as positive labels in
// regression as classification.
constexpr uint32_t REGRESSION_CORRECT_LABEL_RADIUS = 2;

// Batch size to use for processing in query reformulation.
constexpr uint32_t QUERY_REFORMULATION_BATCH_SIZE = 10000;

// Whether to use the Mach Extreme Classification Backend.
constexpr bool USE_MACH = false;

// How many times we hash each entity in UDT Mach Classifier.
constexpr uint32_t MACH_DEFAULT_NUM_REPETITIONS = 7;

// Scaledown factor for output range in UDT Mach Classifier.
constexpr uint32_t MACH_DEFAULT_OUTPUT_RANGE_SCALEDOWN = 25;

// How many output buckets we restrict decoding to for UDT Mach Classifier.
constexpr uint32_t MACH_TOP_K_PER_EVAL_AGGREGATION = 25;

// How many results we're required to decode from the above num buckets for UDT
// Mach Classifier.
constexpr uint32_t MACH_MIN_NUM_EVAL_RESULTS = 5;

// Any less than this number of target classes should cause no scaledown in Mach
constexpr uint32_t MACH_MIN_TARGET_CLASSES = 5000;

// When to switch to using the mach index for sparse inference
constexpr float MACH_SPARSE_INFERENCE_THRESHOLD = 0.2;

// Whether to use the experimental autotune for fully connected layer hash
// tables
constexpr bool EXPERIMENTAL_HIDDEN_LAYER_CONFIG = false;

}  // namespace thirdai::automl::udt::defaults
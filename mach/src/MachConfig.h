#pragma once

#include <bolt/src/nn/model/Model.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::mach {

class MachRetriever;

class MachConfig {
 public:
  static MachConfig make() { return MachConfig(); }

  std::shared_ptr<MachRetriever> build() const;

  MachConfig textCol(const std::string& col) {
    _text_col = col;
    return *this;
  }

  MachConfig idCol(const std::string& col) {
    _id_col = col;
    return *this;
  }

  MachConfig tokenizer(const std::string& tokenizer) {
    _tokenizer = tokenizer;
    return *this;
  }

  MachConfig contextualEncoding(const std::string& encoding) {
    _contextual_encoding = encoding;
    return *this;
  }

  MachConfig lowercase(bool lowercase = true) {
    _lowercase = lowercase;
    return *this;
  }

  MachConfig textFeatureDim(size_t text_feature_dim) {
    _text_feature_dim = text_feature_dim;
    return *this;
  }

  MachConfig embDim(size_t emb_dim) {
    _emb_dim = emb_dim;
    return *this;
  }

  MachConfig nBuckets(size_t n_buckets) {
    _n_buckets = n_buckets;
    return *this;
  }

  MachConfig embBias(bool bias = true) {
    _emb_bias = bias;
    return *this;
  }

  MachConfig outputBias(bool bias = true) {
    _output_bias = bias;
    return *this;
  }

  MachConfig embActivation(const std::string& activation) {
    _emb_act = activation;
    return *this;
  }

  MachConfig outputActivation(const std::string& activation) {
    _output_act = activation;
    return *this;
  }

  MachConfig nHashes(size_t n_hashes) {
    _n_hashes = n_hashes;
    return *this;
  }

  MachConfig machSamplingThreshold(float threshold) {
    _mach_sampling_threshold = threshold;
    return *this;
  }

  MachConfig nBucketsToEval(size_t n_buckets_to_eval) {
    _n_buckets_to_eval = n_buckets_to_eval;
    return *this;
  }

  MachConfig machMemoryParams(size_t max_memory_ids,
                              size_t max_memory_samples_per_id) {
    _max_memory_ids = max_memory_ids;
    _max_memory_samples_per_id = max_memory_samples_per_id;
    return *this;
  }

  MachConfig freezeHashTablesEpoch(uint32_t epoch) {
    _freeze_hash_tables_epoch = epoch;
    return *this;
  }

  data::StatePtr state() const;

  bolt::ModelPtr model() const;

  const std::string& getTextCol() const {
    if (!_text_col) {
      throw std::invalid_argument(
          "Must set the text column of the MachConfig before calling build. "
          "This can be done by calling the `text_col('column name')` method on "
          "the MachConfig.");
    }
    return *_text_col;
  }

  const std::string& getIdCol() const {
    if (!_id_col) {
      throw std::invalid_argument(
          "Must set the id column of the MachConfig before calling build. This "
          "can be done by calling the `id_col('column name')` method on the "
          "MachConfig.");
    }
    return *_id_col;
  }

  data::TextTokenizerPtr textTransformation() const;

  data::MachLabelPtr idToBucketsTransform() const;

  bool usesSoftmax() const { return text::lower(_output_act) == "softmax"; }

  const auto& getFeezeHashTablesEpoch() const {
    return _freeze_hash_tables_epoch;
  }

  float getMachSamplingThreshold() const { return _mach_sampling_threshold; }

  size_t getNBucketsToEval() const { return _n_buckets_to_eval; }

 private:
  // Data parameters
  std::optional<std::string> _text_col = std::nullopt;
  std::optional<std::string> _id_col = std::nullopt;

  // Data processing
  std::string _tokenizer = "char-4";
  std::string _contextual_encoding = "none";
  bool _lowercase = true;

  // Model parameters
  size_t _text_feature_dim = 100000;
  size_t _emb_dim = 512;
  size_t _n_buckets = 50000;

  bool _emb_bias = false;
  bool _output_bias = false;
  std::string _emb_act = "relu";
  std::string _output_act = "softmax";

  // Mach parameters
  size_t _n_hashes = 7;
  float _mach_sampling_threshold = 0.01;
  size_t _n_buckets_to_eval = 25;
  size_t _max_memory_ids = 1000;
  size_t _max_memory_samples_per_id = 10;

  // Other
  std::optional<uint32_t> _freeze_hash_tables_epoch = 1;
};

static constexpr const char* bucket_column = "__buckets__";
static constexpr const char* input_indices_column = "__input_indices__";
static constexpr const char* input_values_column = "__input_values__";

float autotuneSparsity(size_t dim);

}  // namespace thirdai::mach
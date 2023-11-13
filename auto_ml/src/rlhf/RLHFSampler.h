#pragma once

#include <cereal/access.hpp>
#include <data/src/ColumnMap.h>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::automl::udt {

struct RlhfSample {
  std::vector<uint32_t> input_indices;
  std::vector<float> input_values;
  std::vector<uint32_t> mach_buckets;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

class RLHFSampler {
 public:
  RLHFSampler()
      : RLHFSampler("", "", "", "", 0, 0) {
  }  // Required for serializing optional.

  RLHFSampler(std::string input_indices_column, std::string input_values_column,
              std::string doc_id_column, std::string mach_buckets_column,
              size_t max_docs, size_t max_samples_per_doc)
      : _input_indices_column(std::move(input_indices_column)),
        _input_values_column(std::move(input_values_column)),
        _doc_id_column(std::move(doc_id_column)),
        _mach_buckets_column(std::move(mach_buckets_column)),
        _max_docs(max_docs),
        _max_samples_per_doc(max_samples_per_doc),
        _rng(RNG_SEED) {}

  std::optional<data::ColumnMap> balancingSamples(size_t num_samples);

  void addSamples(const data::ColumnMap& columns);

  void addSample(uint32_t doc_id, RlhfSample sample);

  void clear() {
    _samples_per_doc = {};
    _labels = {};
  }

  void removeDoc(uint32_t doc_id) {
    _samples_per_doc.erase(doc_id);
    _labels.erase(doc_id);
  }

 private:
  static constexpr uint32_t RNG_SEED = 7240924;

  std::string _input_indices_column;
  std::string _input_values_column;
  std::string _doc_id_column;
  std::string _mach_buckets_column;
  std::optional<size_t> _input_indices_dim;
  std::optional<size_t> _num_mach_buckets;

  std::unordered_map<uint32_t, std::vector<RlhfSample>> _samples_per_doc;
  std::unordered_set<uint32_t> _labels;

  size_t _max_docs;
  size_t _max_samples_per_doc;

  std::mt19937 _rng{RNG_SEED};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::automl::udt
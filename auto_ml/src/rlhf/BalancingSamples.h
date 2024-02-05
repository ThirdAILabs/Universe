#pragma once

#include "RLHFSampler.h"
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <cstddef>
#include <iterator>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::automl::udt {

struct BalancingSample {
  std::vector<uint32_t> indices;
  std::vector<float> values;
  std::vector<uint32_t> labels;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(indices, values, labels);
  }
};

class BalancingSamples {
 public:
  BalancingSamples() {}  // Required for serializing optional.

  BalancingSamples(std::string indices_col, std::string values_col,
                   std::string labels_col, std::string doc_ids_col,
                   size_t indices_dim, size_t label_dim, size_t max_docs,
                   size_t max_samples_per_doc)
      : _indices_col(std::move(indices_col)),
        _values_col(std::move(values_col)),
        _labels_col(std::move(labels_col)),
        _doc_ids_col(std::move(doc_ids_col)),
        _indices_dim(indices_dim),
        _label_dim(label_dim),
        _max_docs(max_docs),
        _max_samples_per_doc(max_samples_per_doc) {}

  BalancingSamples(std::string indices_col, std::string values_col,
                   std::string labels_col, std::string doc_ids_col,
                   size_t indices_dim, size_t label_dim,
                   const RLHFSampler& sampler);

  explicit BalancingSamples(const ar::Archive& archive);

  data::ColumnMap balancingSamples(size_t num_samples);

  void addSamples(const data::ColumnMap& data);

  void clear() {
    _samples_per_doc = {};
    _doc_ids = {};
  }

  void removeDoc(uint32_t doc_id) {
    _samples_per_doc.erase(doc_id);
    _doc_ids.erase(doc_id);
  }

  ar::ConstArchivePtr toArchive() const;

 private:
  void addSample(uint32_t doc_id, BalancingSample sample);

  static constexpr uint32_t RNG_SEED = 7240924;

  std::string _indices_col;
  std::string _values_col;
  std::string _labels_col;
  std::string _doc_ids_col;

  size_t _indices_dim;
  size_t _label_dim;

  size_t _max_docs;
  size_t _max_samples_per_doc;

  std::unordered_map<uint32_t, std::vector<BalancingSample>> _samples_per_doc;
  std::unordered_set<uint32_t> _doc_ids;

  std::mt19937 _rng{RNG_SEED};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::automl::udt

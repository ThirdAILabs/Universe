#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <cstddef>
#include <iterator>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::automl::udt {

class BalancingSamples;

class RLHFSampler {
  friend class BalancingSamples;

 public:
  RLHFSampler() : RLHFSampler(0, 0) {}  // Required for serializing optional.

  RLHFSampler(size_t max_docs, size_t max_samples_per_doc)
      : _max_docs(max_docs),
        _max_samples_per_doc(max_samples_per_doc),
        _rng(RNG_SEED) {}

  std::vector<std::pair<BoltVector, BoltVector>> balancingSamples(
      size_t num_samples);

  void addSample(uint32_t doc_id, const BoltVector& input,
                 const BoltVector& label);

  void clear() {
    _samples_per_doc = {};
    _doc_ids = {};
  }

  void removeDoc(uint32_t doc_id) {
    _samples_per_doc.erase(doc_id);
    _doc_ids.erase(doc_id);
  }

 private:
  static constexpr uint32_t RNG_SEED = 7240924;

  std::unordered_map<uint32_t, std::vector<std::pair<BoltVector, BoltVector>>>
      _samples_per_doc;
  std::unordered_set<uint32_t> _doc_ids;

  size_t _max_docs;
  size_t _max_samples_per_doc;

  std::mt19937 _rng{RNG_SEED};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::automl::udt

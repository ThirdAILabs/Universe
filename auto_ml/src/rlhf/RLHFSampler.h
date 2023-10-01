#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <cstddef>
#include <iterator>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::automl::udt {

using RlhfSample = std::pair<std::string, std::string>;

class RLHFSampler {
 public:
  RLHFSampler() : RLHFSampler(0, 0) {}  // Required for serializing optional.

  RLHFSampler(size_t max_docs, size_t max_samples_per_doc)
      : _max_docs(max_docs),
        _max_samples_per_doc(max_samples_per_doc),
        _rng(RNG_SEED) {}

  static auto make(size_t max_docs, size_t max_samples_per_doc) {
    return std::make_shared<RLHFSampler>(max_docs, max_samples_per_doc);
  }

  std::vector<RlhfSample> balancingSamples(size_t num_samples) const;

  void addSample(const std::string& doc_id, const RlhfSample& sample);

  void clear() {
    _samples_per_doc = {};
    _doc_ids = {};
  }

  void removeDoc(const std::string& doc_id) {
    _samples_per_doc.erase(doc_id);
    _doc_ids.erase(doc_id);
  }

 private:
  static constexpr uint32_t RNG_SEED = 7240924;

  std::unordered_map<std::string, std::vector<RlhfSample>> _samples_per_doc;
  std::unordered_set<std::string> _doc_ids;

  size_t _max_docs;
  size_t _max_samples_per_doc;

  std::mt19937 _rng{RNG_SEED};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using RLHFSamplerPtr = std::shared_ptr<RLHFSampler>;

}  // namespace thirdai::automl::udt

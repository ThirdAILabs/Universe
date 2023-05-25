#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <unordered_map>

namespace thirdai::automl::udt {

class RLHFSampler {
 public:
  std::pair<std::vector<BoltVector>, std::vector<BoltVector>> balancingSamples(
      uint32_t num_samples);

  void addSample(uint32_t doc_id, const BoltVector& input,
                 const BoltVector& label);

  void clear() { _samples_per_doc = {}; }

  void removeDoc(uint32_t doc_id) { _samples_per_doc.erase(doc_id); }

 private:
  std::unordered_map<uint32_t, std::vector<std::pair<BoltVector, BoltVector>>>
      _samples_per_doc;
  uint32_t _max_samples_per_doc;
  uint32_t _max_docs;
};

}  // namespace thirdai::automl::udt

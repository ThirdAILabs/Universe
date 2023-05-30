#include "RLHFSampler.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

namespace thirdai::automl::udt {

std::pair<std::vector<BoltVector>, std::vector<BoltVector>>
RLHFSampler::balancingSamples(size_t num_samples) {
  std::vector<std::pair<BoltVector, BoltVector>> samples;

  uint32_t full_rounds = num_samples / _samples_per_doc.size();
  for (uint32_t round = 0; round < full_rounds; round++) {
    for (const auto& [_, doc_samples] : _samples_per_doc) {
      std::sample(doc_samples.begin(), doc_samples.end(),
                  std::back_inserter(samples), 1, _rng);
      num_samples--;
    }
  }

  std::vector<uint32_t> docs_to_sample;
  std::sample(_doc_ids.begin(), _doc_ids.end(),
              std::back_inserter(docs_to_sample), num_samples, _rng);
  for (uint32_t doc_id : docs_to_sample) {
    std::sample(_samples_per_doc.at(doc_id).begin(),
                _samples_per_doc.at(doc_id).end(), std::back_inserter(samples),
                1, _rng);
  }

  std::vector<BoltVector> inputs;
  inputs.reserve(samples.size());
  std::vector<BoltVector> labels;
  inputs.reserve(samples.size());
  for (auto& sample : samples) {
    inputs.emplace_back(std::move(sample.first));
    labels.emplace_back(std::move(sample.second));
  }

  return {std::move(inputs), std::move(labels)};
}

void RLHFSampler::addSample(uint32_t doc_id, const BoltVector& input,
                            const BoltVector& label) {
  if (_samples_per_doc.size() >= _max_docs) {
    return;
  }
  if (_samples_per_doc[doc_id].size() < _max_samples_per_doc) {
    _samples_per_doc[doc_id].push_back(std::make_pair(input, label));
    _doc_ids.insert(doc_id);
  } else {
    //  Newer samples have a higher probability of being kept, we can change
    //  this to reservoir sampling if this is an issue.
    std::uniform_int_distribution<> dist(0, _max_samples_per_doc - 1);
    size_t replace = dist(_rng);
    _samples_per_doc[doc_id][replace] = std::make_pair(input, label);
  }
}

template void RLHFSampler::serialize(cereal::BinaryInputArchive& archive);
template void RLHFSampler::serialize(cereal::BinaryOutputArchive& archive);

template <class Archive>
void RLHFSampler::serialize(Archive& archive) {
  archive(_samples_per_doc, _doc_ids, _max_docs, _max_samples_per_doc);
}

}  // namespace thirdai::automl::udt
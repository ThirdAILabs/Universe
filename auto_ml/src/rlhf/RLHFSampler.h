#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <cstddef>
#include <iterator>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::automl::udt {

class RLHFSampler {
 public:
  RLHFSampler(size_t max_docs, size_t max_samples_per_doc)
      : _max_docs(max_docs),
        _max_samples_per_doc(max_samples_per_doc),
        _rng(7240924) {}

  std::pair<std::vector<BoltVector>, std::vector<BoltVector>> balancingSamples(
      size_t num_samples) {
    size_t num_docs_to_sample = std::min(num_samples, _samples_per_doc.size());
    size_t samples_per_doc = static_cast<size_t>(
        std::ceil(static_cast<float>(num_samples)) / num_docs_to_sample);

    std::vector<uint32_t> docs_to_sample;
    std::sample(_doc_ids.begin(), _doc_ids.end(),
                std::back_inserter(docs_to_sample), num_docs_to_sample, _rng);

    std::vector<std::pair<BoltVector, BoltVector>> samples;

    for (uint32_t doc_id : docs_to_sample) {
      std::sample(_samples_per_doc.at(doc_id).begin(),
                  _samples_per_doc.at(doc_id).end(),
                  std::back_inserter(samples), samples_per_doc, _rng);
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

  void addSample(uint32_t doc_id, const BoltVector& input,
                 const BoltVector& label) {
    if (_samples_per_doc.size() >= _max_docs) {
      return;
    }
    if (_samples_per_doc[doc_id].size() < _max_samples_per_doc) {
      _samples_per_doc[doc_id].push_back(std::make_pair(input, label));
      _doc_ids.insert(doc_id);
    } else {
      std::uniform_int_distribution<> dist(0, _max_samples_per_doc - 1);
      size_t replace = dist(_rng);
      _samples_per_doc[doc_id][replace] = std::make_pair(input, label);
    }
  }

  void clear() {
    _samples_per_doc = {};
    _doc_ids = {};
  }

  void removeDoc(uint32_t doc_id) {
    _samples_per_doc.erase(doc_id);
    _doc_ids.erase(doc_id);
  }

 private:
  std::unordered_map<uint32_t, std::vector<std::pair<BoltVector, BoltVector>>>
      _samples_per_doc;
  std::unordered_set<uint32_t> _doc_ids;

  size_t _max_docs;
  size_t _max_samples_per_doc;

  std::mt19937 _rng;
};

}  // namespace thirdai::automl::udt

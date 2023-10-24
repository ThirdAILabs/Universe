#include "RLHFSampler.h"
#include <proto/udt_mach.pb.h>
#include <stdexcept>

namespace thirdai::automl::udt {

RLHFSampler::RLHFSampler(const proto::udt::RlhfSampler& rlhf_sampler)
    : _max_docs(rlhf_sampler.max_docs()),
      _max_samples_per_doc(rlhf_sampler.max_samples_per_doc()) {
  for (const auto& [doc, samples] : rlhf_sampler.samples_per_doc()) {
    if (!samples.samples().empty()) {
      _doc_ids.insert(doc);
    }
    for (const auto& proto_sample : samples.samples()) {
      RlhfSample sample = {
          proto_sample.source(),
          {proto_sample.targets().begin(), proto_sample.targets().end()}};

      _samples_per_doc[doc].emplace_back(std::move(sample));
    }
  }
}

std::vector<RlhfSample> RLHFSampler::balancingSamples(size_t num_samples) {
  if (num_samples == 0) {
    return {};
  }

  if (_samples_per_doc.empty()) {
    throw std::runtime_error(
        "Cannot call associate before training, coldstarting, or introducing "
        "documents.");
  }

  std::vector<RlhfSample> samples;

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

  return samples;
}

void RLHFSampler::addSample(uint32_t doc_id, const RlhfSample& sample) {
  if (_samples_per_doc.size() >= _max_docs) {
    return;
  }
  if (_samples_per_doc[doc_id].size() < _max_samples_per_doc) {
    _samples_per_doc[doc_id].emplace_back(sample);
    _doc_ids.insert(doc_id);
  } else {
    //  Newer samples have a higher probability of being kept, we can change
    //  this to reservoir sampling if this is an issue.
    std::uniform_int_distribution<> dist(0, _max_samples_per_doc - 1);
    size_t replace = dist(_rng);
    _samples_per_doc[doc_id][replace] = sample;
  }
}

proto::udt::RlhfSampler* RLHFSampler::toProto() const {
  auto* rlhf_sampler = new proto::udt::RlhfSampler();

  for (const auto& [doc, samples] : _samples_per_doc) {
    proto::udt::RlhfSamples doc_samples;
    for (const auto& [source, targets] : samples) {
      auto* sample_proto = doc_samples.add_samples();
      sample_proto->set_source(source);
      *sample_proto->mutable_targets() = {targets.begin(), targets.end()};
    }
    rlhf_sampler->mutable_samples_per_doc()->emplace(doc, doc_samples);
  }

  rlhf_sampler->set_max_docs(_max_docs);
  rlhf_sampler->set_max_samples_per_doc(_max_samples_per_doc);

  return rlhf_sampler;
}

}  // namespace thirdai::automl::udt
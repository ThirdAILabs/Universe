#include "RLHFSampler.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace thirdai::automl::udt {

template void RlhfSample::serialize(cereal::BinaryInputArchive& archive);
template void RlhfSample::serialize(cereal::BinaryOutputArchive& archive);

template <class Archive>
void RlhfSample::serialize(Archive& archive) {
  archive(input_indices, input_values, mach_buckets);
}

std::optional<data::ColumnMap> RLHFSampler::balancingSamples(
    size_t num_samples) {
  if (num_samples == 0) {
    return {};
  }

  if (_samples_per_doc.empty()) {
    throw std::runtime_error(
        "Cannot call associate before training, coldstarting, or introducing "
        "documents.");
  }

  std::vector<std::vector<uint32_t>> indices;
  std::vector<std::vector<float>> values;
  std::vector<std::vector<uint32_t>> doc_ids;
  std::vector<std::vector<uint32_t>> buckets;
  indices.reserve(num_samples);
  values.reserve(num_samples);
  buckets.reserve(num_samples);

  uint32_t full_rounds = num_samples / _samples_per_doc.size();
  uint32_t num_extra_round_docs =
      num_samples - full_rounds * _samples_per_doc.size();

  std::unordered_set<uint32_t> extra_round_docs;
  std::sample(_labels.begin(), _labels.end(),
              std::back_inserter(extra_round_docs), num_extra_round_docs, _rng);

  for (const auto& [doc_id, doc_samples] : _samples_per_doc) {
    std::uniform_int_distribution<uint32_t> dist(0, doc_samples.size() - 1);
    uint32_t rounds = full_rounds + extra_round_docs.count(doc_id);
    for (uint32_t round = 0; round < rounds; round++) {
      auto pos = dist(_rng);
      indices.push_back(doc_samples[pos].input_indices);
      values.push_back(doc_samples[pos].input_values);
      buckets.push_back(doc_samples[pos].mach_buckets);
      num_samples--;
    }
  }

  std::unordered_map<std::string, data::ColumnPtr> columns(
      {{_input_indices_column, data::ArrayColumn<uint32_t>::make(
                                   std::move(indices), _input_indices_dim)},
       {_input_values_column,
        data::ArrayColumn<float>::make(std::move(values))},
       {_mach_buckets_column, data::ArrayColumn<uint32_t>::make(
                                  std::move(buckets), _num_mach_buckets)}});

  return data::ColumnMap(std::move(columns));
}

void RLHFSampler::addSamples(const data::ColumnMap& columns) {
  const auto& doc_ids = columns.getArrayColumn<uint32_t>(_doc_id_column);
  const auto& input_indices =
      columns.getArrayColumn<uint32_t>(_input_indices_column);
  const auto& input_values =
      columns.getArrayColumn<float>(_input_values_column);
  const auto& mach_buckets =
      columns.getArrayColumn<uint32_t>(_mach_buckets_column);

  if (_input_indices_dim && _input_indices_dim != input_indices->dim()) {
    throw std::invalid_argument(
        "Input indices dimension is inconsistent with RLHF samples.");
  }
  _input_indices_dim = input_indices->dim();
  if (_num_mach_buckets && _num_mach_buckets != mach_buckets->dim()) {
    throw std::invalid_argument(
        "Number of mach buckets is inconsistent with RLHF samples.");
  }
  _num_mach_buckets = mach_buckets->dim();

  for (size_t i = 0; i < columns.numRows(); i++) {
    if (doc_ids->row(i).size() < 1) {
      continue;
    }
    uint32_t doc_id = doc_ids->row(i)[0];
    automl::udt::RlhfSample sample;
    sample.input_indices = input_indices->row(i).copyToVector();
    sample.input_values = input_values->row(i).copyToVector();
    sample.mach_buckets = mach_buckets->row(i).copyToVector();
    addSample(doc_id, std::move(sample));
  }
}

void RLHFSampler::addSample(uint32_t doc_id, RlhfSample sample) {
  if (_samples_per_doc.size() >= _max_docs) {
    return;
  }
  if (_samples_per_doc[doc_id].size() < _max_samples_per_doc) {
    _samples_per_doc[doc_id].emplace_back(std::move(sample));
    _labels.insert(doc_id);
  } else {
    //  Newer samples have a higher probability of being kept, we can change
    //  this to reservoir sampling if this is an issue.
    std::uniform_int_distribution<> dist(0, _max_samples_per_doc - 1);
    size_t replace = dist(_rng);
    _samples_per_doc[doc_id][replace] = std::move(sample);
  }
}

template void RLHFSampler::serialize(cereal::BinaryInputArchive& archive);
template void RLHFSampler::serialize(cereal::BinaryOutputArchive& archive);

template <class Archive>
void RLHFSampler::serialize(Archive& archive) {
  archive(_input_indices_column, _input_values_column, _doc_id_column,
          _mach_buckets_column, _input_indices_column, _num_mach_buckets,
          _samples_per_doc, _labels, _max_docs, _max_samples_per_doc, _rng);
}

}  // namespace thirdai::automl::udt
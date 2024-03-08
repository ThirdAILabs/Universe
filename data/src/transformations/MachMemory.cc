#include "MachMemory.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace thirdai::data {

std::optional<data::ColumnMap> MachMemory::getSamples(size_t num_samples) {
  if (num_samples == 0) {
    return {};
  }

  if (_id_to_samples.empty()) {
    throw std::runtime_error(
        "Cannot call associate before training, coldstarting, or introducing "
        "documents.");
  }

  std::vector<std::vector<uint32_t>> indices;
  std::vector<std::vector<float>> values;
  std::vector<uint32_t> ids;
  std::vector<std::vector<uint32_t>> buckets;
  indices.reserve(num_samples);
  values.reserve(num_samples);
  ids.reserve(num_samples);
  buckets.reserve(num_samples);

  uint32_t full_rounds = num_samples / _id_to_samples.size();
  uint32_t num_extra_round_docs =
      num_samples - full_rounds * _id_to_samples.size();

  std::unordered_set<uint32_t> extra_round_docs;
  std::sample(_ids.begin(), _ids.end(),
              std::inserter(extra_round_docs, extra_round_docs.begin()),
              num_extra_round_docs, _rng);

  for (const auto& [id, doc_samples] : _id_to_samples) {
    std::uniform_int_distribution<uint32_t> dist(0, doc_samples.size() - 1);
    uint32_t rounds = full_rounds + extra_round_docs.count(id);
    for (uint32_t round = 0; round < rounds; round++) {
      auto pos = dist(_rng);
      indices.push_back(doc_samples[pos].input_indices);
      values.push_back(doc_samples[pos].input_values);
      ids.push_back(id);
      buckets.push_back(doc_samples[pos].mach_buckets);
    }
  }

  std::unordered_map<std::string, data::ColumnPtr> columns(
      {{_input_indices_column, data::ArrayColumn<uint32_t>::make(
                                   std::move(indices), _input_indices_dim)},
       {_input_values_column,
        data::ArrayColumn<float>::make(std::move(values))},
       {_id_column, data::ValueColumn<uint32_t>::make(std::move(ids))},
       {_mach_buckets_column, data::ArrayColumn<uint32_t>::make(
                                  std::move(buckets), _num_mach_buckets)}});

  return data::ColumnMap(std::move(columns));
}

void MachMemory::addSamples(const data::ColumnMap& columns) {
  const auto& ids = columns.getArrayColumn<uint32_t>(_id_column);
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
    if (ids->row(i).size() < 1) {
      continue;
    }
    uint32_t id = ids->row(i)[0];
    MachSample sample;
    sample.input_indices = input_indices->row(i).toVector();
    sample.input_values = input_values->row(i).toVector();
    sample.mach_buckets = mach_buckets->row(i).toVector();
    addSample(id, std::move(sample));
  }
}

void MachMemory::addSample(uint32_t id, MachSample sample) {
  if (_id_to_samples.size() >= _max_ids) {
    return;
  }
  if (_id_to_samples[id].size() < _max_samples_per_id) {
    _id_to_samples[id].emplace_back(std::move(sample));
    _ids.insert(id);
  } else {
    //  Newer samples have a higher probability of being kept, we can change
    //  this to reservoir sampling if this is an issue.
    std::uniform_int_distribution<> dist(0, _max_samples_per_id - 1);
    size_t replace = dist(_rng);
    _id_to_samples[id][replace] = std::move(sample);
  }
}

}  // namespace thirdai::data
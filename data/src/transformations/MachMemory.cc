#include "MachMemory.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <iterator>
#include <limits>
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
       {_id_column, data::ValueColumn<uint32_t>::make(
                        std::move(ids), std::numeric_limits<uint32_t>::max())},
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

ar::ConstArchivePtr MachMemory::toArchive() const {
  auto map = ar::Map::make();

  map->set("input_indices_column", ar::str(_input_indices_column));
  map->set("input_values_column", ar::str(_input_values_column));
  map->set("id_column", ar::str(_id_column));
  map->set("mach_buckets_column", ar::str(_mach_buckets_column));

  if (_input_indices_dim) {
    map->set("input_indices_dim", ar::u64(*_input_indices_dim));
  }
  if (_num_mach_buckets) {
    map->set("num_mach_buckets", ar::u64(*_num_mach_buckets));
  }
  map->set("max_ids", ar::u64(_max_ids));
  map->set("max_samples_per_id", ar::u64(_max_samples_per_id));

  auto balancing_samples_list = ar::List::make();
  for (const auto& [id, samples] : _id_to_samples) {
    auto doc_samples = ar::Map::make();
    doc_samples->set("id", ar::u64(id));

    std::vector<std::vector<uint32_t>> input_indices;
    std::vector<std::vector<float>> input_values;
    std::vector<std::vector<uint32_t>> mach_buckets;

    for (const auto& sample : samples) {
      input_indices.push_back(sample.input_indices);
      input_values.push_back(sample.input_values);
      mach_buckets.push_back(sample.mach_buckets);
    }

    doc_samples->set("input_indices", ar::vecVecU32(std::move(input_indices)));
    doc_samples->set("input_values", ar::vecVecF32(std::move(input_values)));
    doc_samples->set("mach_buckets", ar::vecVecU32(std::move(mach_buckets)));

    balancing_samples_list->append(doc_samples);
  }

  map->set("balancing_samples", balancing_samples_list);

  return map;
}

MachMemory::MachMemory(const ar::Archive& archive)
    : _input_indices_column(archive.str("input_indices_column")),
      _input_values_column(archive.str("input_values_column")),
      _id_column(archive.str("id_column")),
      _mach_buckets_column(archive.str("mach_buckets_column")),
      _input_indices_dim(archive.getOpt<ar::U64>("input_indices_dim")),
      _num_mach_buckets(archive.getOpt<ar::U64>("num_mach_buckets")),
      _max_ids(archive.u64("max_ids")),
      _max_samples_per_id(archive.u64("max_samples_per_id")) {
  for (const auto& samples : archive.get("balancing_samples")->list()) {
    uint64_t id = samples->u64("id");

    const auto& input_indices = samples->getAs<ar::VecVecU32>("input_indices");
    const auto& input_values = samples->getAs<ar::VecVecF32>("input_values");
    const auto& mach_buckets = samples->getAs<ar::VecVecU32>("mach_buckets");

    std::vector<MachSample> mach_samples;
    for (size_t i = 0; i < input_indices.size(); i++) {
      mach_samples.push_back(
          {input_indices.at(i), input_values.at(i), mach_buckets.at(i)});
    }

    _id_to_samples[id] = std::move(mach_samples);
    _ids.insert(id);
  }
}

std::shared_ptr<MachMemory> MachMemory::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<MachMemory>(archive);
}

}  // namespace thirdai::data
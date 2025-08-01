#include "BalancingSamples.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <limits>
#include <stdexcept>

namespace thirdai::automl::udt {

template <typename T>
std::vector<T> vec(const T* data, uint32_t len) {
  return {data, data + len};
}

BalancingSamples::BalancingSamples(std::string indices_col,
                                   std::string values_col,
                                   std::string labels_col,
                                   std::string doc_ids_col, size_t indices_dim,
                                   size_t label_dim, const RLHFSampler& sampler)
    : _indices_col(std::move(indices_col)),
      _values_col(std::move(values_col)),
      _labels_col(std::move(labels_col)),
      _doc_ids_col(std::move(doc_ids_col)),
      _indices_dim(indices_dim),
      _label_dim(label_dim),
      _max_docs(sampler._max_docs),
      _max_samples_per_doc(sampler._max_samples_per_doc),
      _doc_ids(sampler._doc_ids) {
  _samples_per_doc.reserve(sampler._samples_per_doc.size());

  for (const auto& [doc_id, samples] : sampler._samples_per_doc) {
    for (const auto& sample : samples) {
      if (sample.first.isDense() || sample.second.isDense()) {
        throw std::invalid_argument("Cannot convert balancing samples.");
      }
      _samples_per_doc[doc_id].push_back(BalancingSample{
          vec(sample.first.active_neurons, sample.first.len),
          vec(sample.first.activations, sample.first.len),
          vec(sample.second.active_neurons, sample.second.len),
      });
    }
  }
}

data::ColumnMap BalancingSamples::balancingSamples(size_t num_samples) {
  if (num_samples == 0) {
    return data::ColumnMap({});
  }

  if (_samples_per_doc.empty()) {
    throw std::runtime_error(
        "Cannot call associate before training, coldstarting, or introducing "
        "documents.");
  }

  std::vector<uint32_t> doc_ids;
  std::vector<BalancingSample> samples;

  uint32_t full_rounds = num_samples / _samples_per_doc.size();
  for (uint32_t round = 0; round < full_rounds; round++) {
    for (const auto& [doc_id, doc_samples] : _samples_per_doc) {
      std::sample(doc_samples.begin(), doc_samples.end(),
                  std::back_inserter(samples), 1, _rng);
      num_samples--;
      doc_ids.push_back(doc_id);
    }
  }

  std::vector<uint32_t> docs_to_sample;
  std::sample(_doc_ids.begin(), _doc_ids.end(),
              std::back_inserter(docs_to_sample), num_samples, _rng);
  for (uint32_t doc_id : docs_to_sample) {
    std::sample(_samples_per_doc.at(doc_id).begin(),
                _samples_per_doc.at(doc_id).end(), std::back_inserter(samples),
                1, _rng);
    doc_ids.push_back(doc_id);
  }

  std::vector<std::vector<uint32_t>> indices(samples.size());
  std::vector<std::vector<float>> values(samples.size());
  std::vector<std::vector<uint32_t>> labels(samples.size());

  for (size_t i = 0; i < samples.size(); i++) {
    indices[i] = std::move(samples[i].indices);
    values[i] = std::move(samples[i].values);
    labels[i] = std::move(samples[i].labels);
  }

  return createColumnMap(std::move(indices), std::move(values),
                         std::move(labels), std::move(doc_ids));
}

data::ColumnMap BalancingSamples::allBalancingSamples() {
  std::vector<std::vector<uint32_t>> indices;
  std::vector<std::vector<float>> values;
  std::vector<std::vector<uint32_t>> labels;
  std::vector<uint32_t> doc_ids;

  for (const auto& [doc_id, samples] : _samples_per_doc) {
    for (const auto& sample : samples) {
      doc_ids.push_back(doc_id);
      indices.push_back(sample.indices);
      values.push_back(sample.values);
      labels.push_back(sample.labels);
    }
  }

  return createColumnMap(std::move(indices), std::move(values),
                         std::move(labels), std::move(doc_ids));
}

data::ColumnMap BalancingSamples::createColumnMap(
    std::vector<std::vector<uint32_t>>&& indices,
    std::vector<std::vector<float>>&& values,
    std::vector<std::vector<uint32_t>>&& labels,
    std::vector<uint32_t>&& doc_ids) {
  return data::ColumnMap({
      {_indices_col,
       data::ArrayColumn<uint32_t>::make(std::move(indices), _indices_dim)},
      {_values_col, data::ArrayColumn<float>::make(std::move(values))},
      {_labels_col,
       data::ArrayColumn<uint32_t>::make(std::move(labels), _label_dim)},
      {_doc_ids_col,
       data::ValueColumn<uint32_t>::make(std::move(doc_ids),
                                         std::numeric_limits<uint32_t>::max())},
  });
}

void BalancingSamples::addSamples(const data::ColumnMap& data) {
  auto doc_ids = data.getArrayColumn<uint32_t>(_doc_ids_col);
  auto indices = data.getArrayColumn<uint32_t>(_indices_col);
  auto values = data.getArrayColumn<float>(_values_col);
  auto labels = data.getArrayColumn<uint32_t>(_labels_col);

  for (size_t i = 0; i < data.numRows(); i++) {
    BalancingSample sample{
        indices->row(i).toVector(),
        values->row(i).toVector(),
        labels->row(i).toVector(),
    };
    addSample(doc_ids->row(i)[0], std::move(sample));
  }
}

void BalancingSamples::addSample(uint32_t doc_id, BalancingSample sample) {
  if (_samples_per_doc.size() >= _max_docs && !_samples_per_doc.count(doc_id)) {
    return;
  }
  if (_samples_per_doc[doc_id].size() < _max_samples_per_doc) {
    _samples_per_doc[doc_id].emplace_back(std::move(sample));
    _doc_ids.insert(doc_id);
  } else {
    //  Newer samples have a higher probability of being kept, we can change
    //  this to reservoir sampling if this is an issue.
    std::uniform_int_distribution<> dist(0, _max_samples_per_doc - 1);
    size_t replace = dist(_rng);
    _samples_per_doc[doc_id][replace] = std::move(sample);
  }
}

ar::ConstArchivePtr BalancingSamples::toArchive() const {
  auto map = ar::Map::make();

  map->set("indices_col", ar::str(_indices_col));
  map->set("values_col", ar::str(_values_col));
  map->set("labels_col", ar::str(_labels_col));
  map->set("doc_ids_col", ar::str(_doc_ids_col));
  map->set("indices_dim", ar::u64(_indices_dim));
  map->set("label_dim", ar::u64(_label_dim));
  map->set("max_docs", ar::u64(_max_docs));
  map->set("max_samples_per_doc", ar::u64(_max_samples_per_doc));

  auto balancing_samples_list = ar::List::make();
  for (const auto& [doc_id, balancing_samples] : _samples_per_doc) {
    auto doc_samples = ar::Map::make();
    doc_samples->set("doc_id", ar::u64(doc_id));

    std::vector<std::vector<uint32_t>> indices;
    std::vector<std::vector<float>> values;
    std::vector<std::vector<uint32_t>> labels;

    for (const auto& sample : balancing_samples) {
      indices.push_back(sample.indices);
      values.push_back(sample.values);
      labels.push_back(sample.labels);
    }

    doc_samples->set("indices", ar::vecVecU32(std::move(indices)));
    doc_samples->set("values", ar::vecVecF32(std::move(values)));
    doc_samples->set("labels", ar::vecVecU32(std::move(labels)));

    balancing_samples_list->append(doc_samples);
  }

  map->set("balancing_samples", balancing_samples_list);

  return map;
}

BalancingSamples::BalancingSamples(const ar::Archive& archive)
    : _indices_col(archive.str("indices_col")),
      _values_col(archive.str("values_col")),
      _labels_col(archive.str("labels_col")),
      _doc_ids_col(archive.str("doc_ids_col")),
      _indices_dim(archive.u64("indices_dim")),
      _label_dim(archive.u64("label_dim")),
      _max_docs(archive.u64("max_docs")),
      _max_samples_per_doc(archive.u64("max_samples_per_doc")) {
  for (const auto& doc_samples : archive.get("balancing_samples")->list()) {
    uint64_t doc_id = doc_samples->u64("doc_id");

    const auto& indices = doc_samples->getAs<ar::VecVecU32>("indices");
    const auto& values = doc_samples->getAs<ar::VecVecF32>("values");
    const auto& labels = doc_samples->getAs<ar::VecVecU32>("labels");

    std::vector<BalancingSample> samples;
    for (size_t i = 0; i < indices.size(); i++) {
      samples.push_back({indices.at(i), values.at(i), labels.at(i)});
    }

    _samples_per_doc[doc_id] = std::move(samples);
    _doc_ids.insert(doc_id);
  }
}

template void BalancingSamples::serialize(cereal::BinaryInputArchive& archive);
template void BalancingSamples::serialize(cereal::BinaryOutputArchive& archive);

template <class Archive>
void BalancingSamples::serialize(Archive& archive) {
  archive(_indices_col, _values_col, _labels_col, _doc_ids_col, _indices_dim,
          _label_dim, _max_docs, _max_samples_per_doc, _samples_per_doc,
          _doc_ids);
}

}  // namespace thirdai::automl::udt
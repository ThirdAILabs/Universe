#include "BalancingSamples.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <limits>
#include <stdexcept>

namespace thirdai::automl::udt {

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

  return data::ColumnMap({
      {_indices_col,
       data::ArrayColumn<uint32_t>::make(std::move(indices), _indices_dim)},
      {_values_col, data::ArrayColumn<float>::make(std::move(values))},
      {_labels_col,
       data::ArrayColumn<uint32_t>::make(std::move(labels), _labels_dim)},
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

  // The dims are checked when the columns are constructed later, and bolt will
  // check the dims as well. So there is no need to check that the dims match
  // here, we just need to store them.
  _indices_dim = indices->dim();
  _labels_dim = labels->dim();

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

template void BalancingSamples::serialize(cereal::BinaryInputArchive& archive);
template void BalancingSamples::serialize(cereal::BinaryOutputArchive& archive);

template <class Archive>
void BalancingSamples::serialize(Archive& archive) {
  archive(_samples_per_doc, _doc_ids, _max_docs, _max_samples_per_doc);
}

}  // namespace thirdai::automl::udt
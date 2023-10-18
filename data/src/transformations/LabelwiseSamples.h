#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::data {

static ColumnMap keepColumns(ColumnMap&& table,
                             const std::vector<std::string>& columns) {
  std::unordered_map<std::string, thirdai::data::ColumnPtr> new_columns;
  for (const auto& column : columns) {
    new_columns[column] = table.getColumn(column);
  }
  return thirdai::data::ColumnMap(std::move(new_columns));
}

class LabelwiseSamples {
 public:
  LabelwiseSamples(size_t max_labels, size_t max_samples_per_label,
                   std::string label_column, std::vector<std::string> columns)
      : _max_labels(max_labels),
        _max_samples_per_label(max_samples_per_label),
        _label_column(std::move(label_column)),
        _columns(std::move(columns)) {}

  void addSamples(ColumnMap samples) {
    samples = keepColumns(std::move(samples), _columns);

    std::vector<size_t> keep;
    for (uint32_t row = 0; row < samples.numRows(); row++) {
      uint32_t label =
          samples.getArrayColumn<uint32_t>(_label_column)->row(row)[0];

      if (_indices_per_label.size() == _max_labels &&
          !_indices_per_label.count(label)) {
        continue;
      }

      _labels.insert(label);

      if (_indices_per_label[label].size() < _max_samples_per_label) {
        _indices_per_label[label].push_back(_samples->numRows() + keep.size());
        keep.push_back(row);
        continue;
      }

      size_t replace_row = randomSample(_indices_per_label[label]);
      if (replace_row < _samples->numRows()) {
        _samples->setRow(replace_row, samples.row(row));
      } else {
        replace_row = keep.at(replace_row - _samples->numRows());
        samples.setRow(replace_row, samples.row(row));
      }
    }

    samples = samples.permute(keep);
    _num_active_samples += samples.numRows();
    _samples = _samples ? _samples->concat(std::move(samples)) : samples;
  }

  std::optional<ColumnMap> getSamples(uint32_t num_samples) {
    if (!_samples) {
      return std::nullopt;
    }

    std::vector<size_t> indices_to_keep;

    uint32_t full_rounds = num_samples / _indices_per_label.size();

    for (uint32_t round = 0; round < full_rounds; round++) {
      for (auto& [label, indices] : _indices_per_label) {
        indices_to_keep.push_back(randomSample(indices));
      }
    }

    std::vector<uint32_t> docs_to_sample;
    std::sample(_labels.begin(), _labels.end(),
                std::back_inserter(docs_to_sample), num_samples, _rng);
    for (uint32_t label : docs_to_sample) {
      indices_to_keep.push_back(randomSample(_indices_per_label.at(label)));
    }

    return _samples->permute(indices_to_keep);
  }

  void clear() {
    _num_active_samples = 0;
    _samples = std::nullopt;
    _indices_per_label.clear();
    _labels = {};
  }

  void removeEntity(uint32_t doc_id) {
    _num_active_samples -= _indices_per_label.at(doc_id).size();
    _indices_per_label.erase(doc_id);
    _labels.erase(doc_id);

    if (_num_active_samples == 0) {
      _samples = std::nullopt;
      return;
    }

    if (_num_active_samples < _samples->numRows()) {
      std::vector<size_t> keep;
      keep.reserve(_num_active_samples);
      for (auto& [_, indices] : _indices_per_label) {
        size_t new_first_index = keep.size();
        keep.insert(keep.end(), indices.begin(), indices.end());
        std::iota(indices.begin(), indices.end(), new_first_index);
      }
      _samples = _samples->permute(keep);
    }
  }

 private:
  size_t randomSample(const std::vector<size_t>& indices) {
    std::uniform_int_distribution<> dist(0, indices.size() - 1);
    return indices[dist(_rng)];
  }

  static constexpr uint32_t RNG_SEED = 7240924;

  size_t _max_labels;
  size_t _max_samples_per_label;
  std::string _label_column;
  std::vector<std::string> _columns;

  size_t _num_active_samples = 0;
  std::optional<ColumnMap> _samples;
  std::unordered_set<uint32_t> _labels;
  std::unordered_map<uint32_t, std::vector<size_t>> _indices_per_label;
  std::mt19937 _rng{RNG_SEED};
};

}  // namespace thirdai::data
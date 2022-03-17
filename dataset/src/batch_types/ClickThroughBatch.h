#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Factory.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace thirdai::dataset {

static constexpr int CATEGORICAL_FEATURE_BASE = 10;

class ClickThroughBatch {
  friend class ClickThroughBatchFactory;

 public:
  explicit ClickThroughBatch(uint64_t start_id)
      : _batch_size(0), _start_id(start_id) {}

  const bolt::BoltVector& operator[](uint32_t i) const {
    return _dense_features[i];
  };

  bolt::BoltVector& operator[](uint32_t i) { return _dense_features[i]; };

  const bolt::BoltVector& at(uint32_t i) const { return _dense_features.at(i); }

  const bolt::BoltVector& labels(uint32_t i) const { return _labels[i]; }

  const std::vector<uint32_t>& categoricalFeatures(uint32_t i) const {
    return _categorical_features[i];
  }

  uint64_t id(uint32_t i) const { return _start_id + i; }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  std::vector<bolt::BoltVector> _dense_features;
  std::vector<std::vector<uint32_t>> _categorical_features;
  std::vector<bolt::BoltVector> _labels;
  uint32_t _batch_size;
  uint64_t _start_id;
};

class ClickThroughBatchFactory : public Factory<ClickThroughBatch> {
 public:
  ClickThroughBatchFactory(uint32_t num_dense_features,
                           uint32_t num_categorical_features,
                           bool sparse_labels)
      : _num_dense_features(num_dense_features),
        _num_categorical_features(num_categorical_features),
        _sparse_labels(sparse_labels) {}

  ClickThroughBatch parse(std::ifstream& file, uint32_t target_batch_size,
                          uint64_t start_id) override {
    ClickThroughBatch batch(start_id);

    std::string line;
    while (batch._batch_size < target_batch_size && std::getline(file, line)) {
      const char* start = line.c_str();
      char* end;

      uint32_t label = std::strtol(start, &end, 10);
      if (_sparse_labels) {
        bolt::BoltVector label_vec(1, false, false);
        label_vec.active_neurons[0] = label;
        label_vec.activations[0] = 1.0;
        batch._labels.push_back(std::move(label_vec));
      } else {
        bolt::BoltVector label_vec(1, true, false);
        label_vec.activations[0] = label;
        batch._labels.push_back(std::move(label_vec));
      }

      start = end + 1;

      bolt::BoltVector vec(_num_dense_features, true, false);
      for (uint32_t d = 0; d < _num_dense_features; d++) {
        float feature;
        if (*start == '\t') {
          // There could be an empty field if the value is missing
          feature = 0;
          start = start + 1;
        } else {
          feature = std::strtof(start, &end);
          start = end + 1;
        }
        vec.activations[d] = feature;
      }
      batch._dense_features.push_back(std::move(vec));

      std::vector<uint32_t> categorical(_num_categorical_features);
      for (uint32_t c = 0; c < _num_categorical_features; c++) {
        uint32_t feature;
        if (*start == '\t') {
          // There could be an empty field if the value is missing
          feature = 0;
          start = start + 1;
        } else {
          feature = std::strtol(start, &end, CATEGORICAL_FEATURE_BASE);
          start = end + 1;
        }
        categorical[c] = feature;
      }
      batch._categorical_features.push_back(std::move(categorical));

      batch._batch_size++;
    }

    return batch;
  }

 private:
  uint32_t _num_dense_features, _num_categorical_features;
  bool _sparse_labels;
};

}  // namespace thirdai::dataset
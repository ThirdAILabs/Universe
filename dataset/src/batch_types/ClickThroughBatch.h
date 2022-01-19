#pragma once

#include <dataset/src/Factory.h>
#include <dataset/src/Vectors.h>
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

  const DenseVector& operator[](uint32_t i) const {
    return _dense_features[i];
  };

  const DenseVector& at(uint32_t i) const { return _dense_features.at(i); }

  uint32_t label(uint32_t i) const { return _labels[i]; }

  const std::vector<uint32_t>& categoricalFeatures(uint32_t i) const {
    return _categorical_features[i];
  }

  uint64_t id(uint32_t i) const { return _start_id + i; }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  std::vector<DenseVector> _dense_features;
  std::vector<std::vector<uint32_t>> _categorical_features;
  std::vector<uint32_t> _labels;
  uint32_t _batch_size;
  uint64_t _start_id;
};

class ClickThroughBatchFactory : public Factory<ClickThroughBatch> {
 public:
  ClickThroughBatchFactory(uint32_t num_dense_features,
                           uint32_t num_categorical_features)
      : _num_dense_features(num_dense_features),
        _num_categorical_features(num_categorical_features) {}

  ClickThroughBatch parse(std::ifstream& file, uint32_t target_batch_size,
                          uint64_t start_id) override {
    ClickThroughBatch batch(start_id);

    std::string line;
    while (batch._batch_size < target_batch_size && std::getline(file, line)) {
      const char* start = line.c_str();
      char* end;

      uint32_t label = std::strtol(start, &end, 10);
      batch._labels.push_back(label);

      start = end + 1;

      DenseVector vec(_num_dense_features);
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
        vec._values[d] = feature;
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
};

}  // namespace thirdai::dataset
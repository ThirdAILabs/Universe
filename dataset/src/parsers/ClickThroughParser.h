#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <fstream>

namespace thirdai::dataset {

class ClickThroughParser {
 public:
  explicit ClickThroughParser(uint32_t num_dense_features,
                              uint32_t num_categorical_features,
                              bool sparse_labels)
      : _num_dense_features(num_dense_features),
        _num_categorical_features(num_categorical_features),
        _sparse_labels(sparse_labels) {}

  std::pair<ClickThroughBatch, bolt::BoltBatch> parseBatch(
      uint32_t target_batch_size, std::ifstream& file) const {
    ClickThroughBatch batch;
    std::vector<bolt::BoltVector> labels;

    uint32_t curr_batch_size = 0;

    std::string line;
    while (curr_batch_size < target_batch_size && std::getline(file, line)) {
      const char* start = line.c_str();
      char* end;

      uint32_t label = std::strtol(start, &end, 10);
      if (_sparse_labels) {
        bolt::BoltVector label_vec(1, false, false);
        label_vec.active_neurons[0] = label;
        label_vec.activations[0] = 1.0;
        labels.push_back(std::move(label_vec));
      } else {
        bolt::BoltVector label_vec(1, true, false);
        label_vec.activations[0] = label;
        labels.push_back(std::move(label_vec));
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

      curr_batch_size++;
    }

    return std::pair<ClickThroughBatch, bolt::BoltBatch>{std::move(batch),
                                                         std::move(labels)};
  }

 private:
  uint32_t _num_dense_features, _num_categorical_features;
  bool _sparse_labels;
};

}  // namespace thirdai::dataset

#pragma once

#include "ProcessorUtils.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

class ClickThroughBatchProcessor final
    : public BatchProcessor<bolt::BoltBatch, BoltTokenBatch, bolt::BoltBatch> {
 public:
  ClickThroughBatchProcessor(uint32_t num_dense_features,
                             uint32_t max_num_categorical_features,
                             char delimiter = '\t')
      : _num_dense_features(num_dense_features),
        _expected_num_cols(num_dense_features + max_num_categorical_features +
                           1),
        _delimiter(delimiter) {}

  std::tuple<bolt::BoltBatch, BoltTokenBatch, bolt::BoltBatch> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<bolt::BoltVector> dense_inputs(rows.size());
    std::vector<std::vector<uint32_t>> token_inputs(rows.size());
    std::vector<bolt::BoltVector> labels(rows.size());

    for (uint32_t i = 0; i < rows.size(); i++) {
      auto [data_vec, tokens, label] = processRow(rows[i]);

      dense_inputs[i] = std::move(data_vec);
      token_inputs[i] = std::move(tokens);
      labels[i] = std::move(label);
    }

    return {bolt::BoltBatch(std::move(dense_inputs)),
            BoltTokenBatch(std::move(token_inputs)),
            bolt::BoltBatch(std::move(labels))};
  }

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

 private:
  std::tuple<bolt::BoltVector, std::vector<uint32_t>, bolt::BoltVector>
  processRow(const std::string& row) const {
    auto cols = ProcessorUtils::parseCsvRow(row, _delimiter);

    if (cols.size() <= _num_dense_features + 1) {
      throw std::invalid_argument(
          "Expected at least " + std::to_string(_expected_num_cols) +
          " columns in click through dataset, received line with " +
          std::to_string(cols.size()) + " columns.");
    }

    auto label = getLabelVector(/* label_str= */ cols[0]);

    std::vector<float> dense_features;
    uint32_t feature_idx = 1;
    for (; feature_idx < _num_dense_features + 1; feature_idx++) {
      if (cols[feature_idx].empty()) {
        dense_features.push_back(0.0);
        continue;
      }
      char* end;
      float val = std::strtof(cols[feature_idx].data(), &end);
      dense_features.push_back(val);
    }

    std::vector<uint32_t> categorical_features;
    for (; feature_idx < cols.size(); feature_idx++) {
      if (cols[feature_idx].empty()) {
        categorical_features.push_back(0);
        continue;
      }
      char* end;
      uint32_t val =
          std::strtoul(cols[feature_idx].data(), &end, /* base= */ 10);
      categorical_features.push_back(val);
    }

    return {bolt::BoltVector::makeDenseVector(dense_features),
            std::move(categorical_features), std::move(label)};
  }

  static bolt::BoltVector getLabelVector(const std::string_view& label_str) {
    char* end;
    uint32_t label = std::strtol(label_str.data(), &end, 10);
    bolt::BoltVector label_vec(1, /* is_dense= */ false,
                               /* has_gradient= */ false);
    label_vec.active_neurons[0] = label;
    label_vec.activations[0] = 1.0;
    return label_vec;
  }

  uint32_t _num_dense_features, _expected_num_cols;
  char _delimiter;
};

}  // namespace thirdai::dataset
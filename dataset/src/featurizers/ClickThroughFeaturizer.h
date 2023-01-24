#pragma once

#include "ProcessorUtils.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

class ClickThroughFeaturizer final : public Featurizer {
 public:
  ClickThroughFeaturizer(uint32_t num_dense_features,
                         uint32_t max_num_categorical_features,
                         char delimiter = '\t')
      : _num_dense_features(num_dense_features),
        _expected_num_cols(num_dense_features + max_num_categorical_features +
                           1),
        _delimiter(delimiter) {}

  std::vector<BoltBatch> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<BoltVector> dense_inputs(rows.size());
    std::vector<BoltVector> token_inputs(rows.size());
    std::vector<BoltVector> labels(rows.size());

    for (uint32_t i = 0; i < rows.size(); i++) {
      auto [data_vec, tokens, label] = processRow(rows[i]);

      dense_inputs[i] = std::move(data_vec);
      token_inputs[i] = std::move(tokens);
      labels[i] = std::move(label);
    }

    return {BoltBatch(std::move(dense_inputs)),
            BoltBatch(std::move(token_inputs)), BoltBatch(std::move(labels))};
  }

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

 private:
  std::tuple<BoltVector, BoltVector, BoltVector> processRow(
      const std::string& row) const {
    auto cols = ProcessorUtils::parseCsvRow(row, _delimiter);

    if (cols.size() <= _num_dense_features + 1) {
      throw std::invalid_argument(
          "Expected at least " + std::to_string(_expected_num_cols) +
          " columns in click through dataset, received line with " +
          std::to_string(cols.size()) + " columns.");
    }

    auto label = getLabelVector(/* label_str= */ cols[0]);

    std::vector<float> dense_features;

    for (uint32_t feature_idx = 1; feature_idx < _num_dense_features + 1;
         feature_idx++) {
      if (cols[feature_idx].empty()) {
        dense_features.push_back(0.0);
        continue;
      }
      char* end;
      float val = std::strtof(cols[feature_idx].data(), &end);
      dense_features.push_back(std::log(val + 1));
    }

    // Its _num_dense_features + 1 because the label is the first column.
    uint32_t index_of_first_categorical_feature = _num_dense_features + 1;
    BoltVector categorical_features(
        cols.size() - index_of_first_categorical_feature,
        /* is_dense= */ false,
        /* has_gradient= */ false);

    for (uint32_t feature_idx = 0;
         feature_idx < cols.size() - index_of_first_categorical_feature;
         feature_idx++) {
      if (cols[index_of_first_categorical_feature + feature_idx].empty()) {
        categorical_features.active_neurons[feature_idx] = 0;
        categorical_features.activations[feature_idx] = 0;
        continue;
      }
      char* end;
      uint32_t val = std::strtoul(
          cols[index_of_first_categorical_feature + feature_idx].data(), &end,
          /* base= */ 10);
      categorical_features.active_neurons[feature_idx] = val;
      categorical_features.activations[feature_idx] = 1.0;
    }

    return {BoltVector::makeDenseVector(dense_features),
            std::move(categorical_features), std::move(label)};
  }

  static BoltVector getLabelVector(const std::string_view& label_str) {
    char* end;
    uint32_t label = std::strtol(label_str.data(), &end, 10);
    return BoltVector::singleElementSparseVector(label);
  }

  uint32_t _num_dense_features, _expected_num_cols;
  char _delimiter;
};

}  // namespace thirdai::dataset
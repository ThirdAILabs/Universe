#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Factory.h>
#include <iostream>
#include <vector>

namespace thirdai::dataset {

class ClickThroughParser;

static constexpr int CATEGORICAL_FEATURE_BASE = 10;

/**
 * The click through batch is the format used by the DLRM model. It expects a
 * label followed by a series of values to be interpreted as a dense vector, and
 * finally a series of categorical features.
 */
class ClickThroughBatch {
  friend class ClickThroughParser;

 public:
  ClickThroughBatch() {}

  ClickThroughBatch(std::vector<bolt::BoltVector>&& dense_features,
                    std::vector<std::vector<uint32_t>>&& categorical_features)
      : _dense_features(std::move(dense_features)),
        _categorical_features(std::move(categorical_features)) {}

  const bolt::BoltVector& operator[](uint32_t i) const {
    return _dense_features[i];
  };

  bolt::BoltVector& operator[](uint32_t i) { return _dense_features[i]; };

  const bolt::BoltVector& at(uint32_t i) const { return _dense_features.at(i); }

  const std::vector<uint32_t>& categoricalFeatures(uint32_t i) const {
    return _categorical_features[i];
  }

  uint32_t getBatchSize() const { return _dense_features.size(); }

 private:
  std::vector<bolt::BoltVector> _dense_features;
  std::vector<std::vector<uint32_t>> _categorical_features;
};

}  // namespace thirdai::dataset
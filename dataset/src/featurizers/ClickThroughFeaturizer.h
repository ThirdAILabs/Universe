#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>

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

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) final;

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  size_t getNumDatasets() final { return 3; }

 private:
  std::tuple<BoltVector, BoltVector, BoltVector> processRow(
      const std::string& row) const;

  static BoltVector getLabelVector(const std::string_view& label_str);

  uint32_t _num_dense_features, _expected_num_cols;
  char _delimiter;
};

}  // namespace thirdai::dataset
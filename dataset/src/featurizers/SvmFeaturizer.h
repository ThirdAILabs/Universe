#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>

namespace thirdai::dataset {

class SvmFeaturizer final : public Featurizer {
 public:
  explicit SvmFeaturizer(bool softmax_for_multiclass = true)
      : _softmax_for_multiclass(softmax_for_multiclass) {}

  bool expectsHeader() const final { return false; }

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) final;

  void processHeader(const std::string& header) final { (void)header; }

  size_t getNumDatasets() final { return 2; }

 private:
  std::pair<BoltVector, BoltVector> processRow(const std::string& line) const;

  bool _softmax_for_multiclass;
};

}  // namespace thirdai::dataset

#pragma once

#include <dataset/src/blocks/TabularBlocks.h>
#include <dataset/src/bolt_datasets/BatchProcessor.h>

namespace thirdai::dataset {

class TabularMetadataProcessor : public ComputeBatchProcessor {
  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {}

  std::pair<bolt::BoltVector, bolt::BoltVector> processRow(
      const std::string& row) final {}
};

}  // namespace thirdai::dataset

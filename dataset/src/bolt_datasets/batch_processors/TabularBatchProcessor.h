#pragma once

#include "GenericBatchProcessor.h"

namespace thirdai::dataset {

class TabularBatchProcessor : public GenericBatchProcessor {
 public:
  TabularBatchProcessor(std::vector<std::shared_ptr<Block>> input_blocks,
                        std::vector<std::shared_ptr<Block>> label_blocks,
                        std::shared_ptr<dataset::TabularMetadata> metadata)
      : GenericBatchProcessor(std::move(input_blocks), std::move(label_blocks),
                              /* expects_header */ true),
        _metadata(metadata) {}

  void processHeader(const std::string& header) override {
    std::vector<std::string_view> actualColumns = parseCsvRow(header, ',');
    std::vector<std::string> expectedColumns = _metadata->getColumnNames();
    if (actualColumns.size() != expectedColumns.size()) {
      throw std::invalid_argument(
          "Expected " + std::to_string(expectedColumns.size()) +
          " columns but received " + std::to_string(actualColumns.size()) +
          " columns.");
    }
    for (uint32_t col = 0; col < expectedColumns.size(); col++) {
      if (actualColumns[col] != expectedColumns[col]) {
        throw std::invalid_argument("Expected column '" + expectedColumns[col] +
                                    "' but received column '" +
                                    std::string(actualColumns[col]) + ".'");
      }
    }
  }

  std::shared_ptr<dataset::TabularMetadata> _metadata;
};

}  // namespace thirdai::dataset
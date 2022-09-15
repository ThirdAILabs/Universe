#pragma once

#include "BlockInterface.h"
#include <dataset/src/batch_processors/NewTabularMetadataProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exception>

namespace thirdai::dataset {
/**
 * Given some metadata about a tabular dataset, assign unique categories to
 * columns and compute pairgrams of the categories.
 */
class TabularPairGram : public Block {
 public:
  TabularPairGram(std::shared_ptr<TabularMetadata>& metadata,
                  uint32_t output_range)
      : _metadata(metadata), _output_range(output_range) {}

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final {
    return _metadata->column_metadata;
  };

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> unigram_hashes;
    for (uint32_t col = 0; col < input_row.size(); col++) {
      std::string str_val(input_row[col]);
      auto col_processor = _metadata->column_metadata[col];
      if (!col_processor.isLabel()) {
        uint32_t unigram = col_processor.getUnigram()
      }
    }
  }

 private:
  std::shared_ptr<TabularMetadata> _metadata;
  uint32_t _output_range;
  // one additional bin is reserved for empty values
  uint32_t _num_non_empty_bins = 10;
};

}  // namespace thirdai::dataset

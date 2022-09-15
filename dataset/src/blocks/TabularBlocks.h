#pragma once

#include "BlockInterface.h"
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exception>

namespace thirdai::dataset {

/**
 * Given some metadata about tabular columns, assign unique categories to
 * columns and compute pairgrams of the categories.
 */
class TabularPairGram : public Block {
 public:
  TabularPairGram(std::vector<TabularColumn> column_metadata,
                  uint32_t output_range)
      : _column_metadata(column_metadata), _output_range(output_range) {}

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _column_metadata.size(); };

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> unigram_hashes;
    for (const auto& tabular_column : _column_metadata) {
      uint32_t col_num = tabular_column.col_num;
      if (col_num >= input_row.size()) {
        return std::make_exception_ptr(std::invalid_argument(
            "TabularColumn operates on a column not present in the input "
            "row. Operates on column " +
            std::to_string(col_num) + " but processed " +
            std::to_string(input_row.size()) + " columns."));
      }
      std::string str_val(input_row[col_num]);
      if (!tabular_column.isLabel()) {
        uint32_t unigram;
        if (auto err = col_processor.computeUnigram(str_val, unigram)) {
          return err;
        }
        unigram_hashes.push_back(unigram);
      }

      std::vector<uint32_t> pairgram_hashes =
          TextEncodingUtils::computeRawPairgramsFromUnigrams(unigram_hashes,
                                                             _output_range);

      TextEncodingUtils::sumRepeatedIndices(
          pairgram_hashes, /* base_value = */ 1.0,
          [&](uint32_t pairgram, float value) {
            vec.addSparseFeatureToSegment(pairgram, value);
          });

      return nullptr;
    }
  }

 private:
  std::vector<TabularColumn> _column_metadata;
  uint32_t _output_range;
};

}  // namespace thirdai::dataset

#pragma once

#include "BlockInterface.h"
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
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

  uint32_t expectedNumColumns() const final { return _metadata->numColumns(); };

  ResponsibleColumnAndInputKey explainFeature(
      uint32_t index_within_block,
      std::optional<std::unordered_map<uint32_t, std::string>> num_to_name,
      std::vector<std::string_view> /*columnar_sample*/) const final {
    (void)index_within_block;
    (void)num_to_name;
    throw std::invalid_argument("not yet implemented in tabular block!");
  }

 protected:
  // TODO(david) We should always include all unigrams but if the number of
  // columns is too large, this processing time becomes slow. One idea is to
  // cap the number of pairgrams at a certain threshold by selecting random
  // pairs of columns to pairgram together.
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> unigram_hashes;
    for (uint32_t col = 0; col < input_row.size(); col++) {
      std::string str_val(input_row[col]);
      switch (_metadata->getColType(col)) {
        case TabularDataType::Numeric: {
          std::exception_ptr err;
          uint32_t unigram = _metadata->getNumericHashValue(col, str_val, err);
          if (err) {
            return err;
          }
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Categorical: {
          uint32_t unigram = _metadata->getStringHashValue(str_val, col);
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Label:
          break;
      }
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

 private:
  std::shared_ptr<TabularMetadata> _metadata;
  uint32_t _output_range;
};

}  // namespace thirdai::dataset
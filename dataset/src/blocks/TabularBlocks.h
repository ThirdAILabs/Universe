#pragma once

#include "BlockInterface.h"
#include <dataset/src/bolt_datasets/batch_processors/PairgramHasher.h>
#include <dataset/src/bolt_datasets/batch_processors/TabularMetadataProcessor.h>

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

 protected:
  // TODO(david) We should always include all unigrams but if the number of
  // columns is too large, this processing time becomes slow. One idea is to cap
  // the number of pairgrams at a certain threshold by selecting random pairs of
  // columns to pairgram together.
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec,
                    std::exception_ptr exception_ptr) final {
    std::vector<uint32_t> unigram_hashes;
    for (uint32_t col = 0; col < input_row.size(); col++) {
      std::string str_val(input_row[col]);
      switch (_metadata->getColType(col)) {
        case TabularDataType::Numeric: {
          uint32_t unigram =
              _metadata->getNumericHashValue(col, str_val, exception_ptr);
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

    // TODO(david) optimize/benchmark pairgram computation?
    std::unordered_map<uint32_t, uint32_t> pairgram_hashes =
        PairgramHasher::computeRawPairgramsFromUnigrams(unigram_hashes,
                                                        _output_range);
    for (auto& entry : pairgram_hashes) {
      vec.addSparseFeatureToSegment(entry.first, 1.0);
    }
  }

 private:
  std::shared_ptr<TabularMetadata> _metadata;
  uint32_t _output_range;
};

}  // namespace thirdai::dataset

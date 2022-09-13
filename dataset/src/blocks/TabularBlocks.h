#pragma once

#include "BlockInterface.h"
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exception>
#include <string>
#include <unordered_map>

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

  std::pair<std::string, std::string> explainIndex(
      uint32_t index,
      std::optional<std::unordered_map<uint32_t, std::string>> num_to_name)
      final {
    (void)num_to_name;
    auto [col_1_num, col_2_num] = getColNumColNum(index);
    std::string col_name, word_responsible;
    if (col_1_num == col_2_num) {
      col_name = _metadata->getColNameFromNum(col_1_num);
      word_responsible = _col_num_to_col_value[col_1_num];
    } else {
      col_name = _metadata->getColNameFromNum(col_1_num) +
                 _metadata->getColNameFromNum(col_2_num);
      word_responsible =
          _col_num_to_col_value[col_1_num] + _col_num_to_col_value[col_2_num];
    }
    return std::make_pair(col_name, word_responsible);
  }

 protected:
  // TODO(david) We should always include all unigrams but if the number of
  // columns is too large, this processing time becomes slow. One idea is to
  // cap the number of pairgrams at a certain threshold by selecting random
  // pairs of columns to pairgram together.
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec, bool store_map) final {
    std::unordered_map<uint32_t, uint32_t> hash_to_col_num;
    std::vector<uint32_t> unigram_hashes;
    for (uint32_t col = 0; col < input_row.size(); col++) {
      std::string str_val(input_row[col]);
      switch (_metadata->getColType(col)) {
        case TabularDataType::Numeric: {
          std::exception_ptr err;
          uint32_t unigram = _metadata->getNumericHashValue(col, str_val, err);
          if (store_map) {
            hash_to_col_num[unigram] = col;
            _col_num_to_col_value[col] = str_val;
          }
          if (err) {
            return err;
          }
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Categorical: {
          uint32_t unigram = _metadata->getStringHashValue(str_val, col);
          if (store_map) {
            hash_to_col_num[unigram] = col;
            _col_num_to_col_value[col] = str_val;
          }
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

    if (store_map) {
      for (uint32_t i = 0; i < unigram_hashes.size(); i++) {
        for (uint32_t j = 0; j <= i; j++) {
          _hash_to_col_num_col_num[pairgram_hashes[((i * (i + 1)) / 2) + i +
                                                   j]] = {
              hash_to_col_num[unigram_hashes[i]],
              hash_to_col_num[unigram_hashes[j]]};
        }
      }
    }

    TextEncodingUtils::sumRepeatedIndices(
        pairgram_hashes, /* base_value = */ 1.0,
        [&](uint32_t pairgram, float value) {
          vec.addSparseFeatureToSegment(pairgram, value);
        });

    return nullptr;
  }

 private:
  std::pair<uint32_t, uint32_t> getColNumColNum(uint32_t index) {
    return _hash_to_col_num_col_num[index];
  }
  std::shared_ptr<TabularMetadata> _metadata;
  uint32_t _output_range;

  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>>
      _hash_to_col_num_col_num;

  std::unordered_map<uint32_t, std::string> _col_num_to_col_value;
};

}  // namespace thirdai::dataset

#pragma once

#include "BlockInterface.h"
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::dataset {

/**
 * @brief Given some metadata about a tabular dataset, assign unique categories
 * to columns and compute pairgrams of the categories.
 *
 * TODO(david): add a TabularBinningStrategy class to try out different methods
 */
class TabularPairGram : public Block {
 public:
  TabularPairGram(TabularMetadataPtr metadata, uint32_t output_range)
      : _metadata(std::move(metadata)), _output_range(output_range) {}

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _metadata->numColumns(); };

  Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& input_row) final {
    auto hash_to_word_map = fillHashesMap(input_row);
    auto column_num_pair = hash_to_word_map[index_within_block];
    std::string reason;
    std::string column_names;
    if (column_num_pair.first != column_num_pair.second) {
      reason = "col: " + std::to_string(column_num_pair.first) + " - " +
               std::string(input_row[column_num_pair.first]) +
               ", col: " + std::to_string(column_num_pair.second) + " - " +
               std::string(input_row[column_num_pair.second]);
      column_names = _metadata->getColumnName(column_num_pair.first) + "," +
                     _metadata->getColumnName(column_num_pair.second);
    } else {
      reason = "col: " + std::to_string(column_num_pair.first) + " - " +
               std::string(input_row[column_num_pair.first]);
      column_names = _metadata->getColumnName(column_num_pair.first);
    }

    return {reason, column_names};
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
    std::exception_ptr err = getUnigramHashes(input_row, unigram_hashes);
    if (err) {
      return err;
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

  std::exception_ptr getUnigramHashes(
      const std::vector<std::string_view>& input_row,
      std::vector<uint32_t>& unigram_hashes) {
    for (uint32_t col = 0; col < input_row.size(); col++) {
      std::string str_val(input_row[col]);
      switch (_metadata->colType(col)) {
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
    return nullptr;
  }

  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> fillHashesMap(
      const std::vector<std::string_view>& input_row) {
    std::vector<uint32_t> unigram_hashes;
    std::exception_ptr err = getUnigramHashes(input_row, unigram_hashes);
    if (err) {
      std::rethrow_exception(err);
    }

    auto hashes_map = TextEncodingUtils::computePairgramsColNumMapFromUnigrams(
        unigram_hashes, _output_range, _metadata->getLabelCol());

    return hashes_map;
  }

 private:
  TabularMetadataPtr _metadata;
  uint32_t _output_range;
};

using TabularPairGramPtr = std::shared_ptr<TabularPairGram>;

}  // namespace thirdai::dataset

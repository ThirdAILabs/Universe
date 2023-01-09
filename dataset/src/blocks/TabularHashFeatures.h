#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <hashing/src/UniversalHash.h>
#include <dataset/src/utils/TabularMetadata.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exception>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::dataset {

/**
 * @brief Given some metadata about a tabular dataset, assign unique categories
 * to columns and compute either unigrams or pairgrams of the categories
 * depending on the "with_pairgrams" flag.
 */
class TabularHashFeatures : public Block {
 public:
  TabularHashFeatures(TabularMetadataPtr metadata, uint32_t output_range,
                      bool with_pairgrams = true)
      : _metadata(std::move(metadata)),
        _output_range(output_range),
        _with_pairgrams(with_pairgrams) {}

  struct Token {
    uint32_t token;
    uint32_t first_column;
    uint32_t second_column;
  };

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _metadata->numColumns(); };

  Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& input_row) final {
    auto [col_num_1, col_num_2] =
        getColumnNumbersForIndex(input_row, index_within_block);

    if (col_num_1 == col_num_2) {
      return {_metadata->getColumnName(col_num_1),
              std::string(input_row[col_num_1])};
    }

    auto column_name = _metadata->getColumnName(col_num_1) + "," +
                       _metadata->getColumnName(col_num_2);
    auto keyword = std::string(input_row[col_num_1]) + "," +
                   std::string(input_row[col_num_2]);

    return {column_name, keyword};
  }

 protected:
  // TODO(david) We should always include all unigrams but if the number of
  // columns is too large, this processing time becomes slow. One idea is to
  // cap the number of pairgrams at a certain threshold by selecting random
  // pairs of columns to pairgram together.
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    std::unordered_map<uint32_t, uint32_t> unigram_to_col_num;
    std::vector<uint32_t> tokens;
    if (auto e = forEachOutputToken(
            input_row, [&](Token token) { tokens.push_back(token.token); })) {
      return e;
    };

    TextEncodingUtils::sumRepeatedIndices(
        tokens, /* base_value = */ 1.0, [&](uint32_t pairgram, float value) {
          vec.addSparseFeatureToSegment(pairgram, value);
        });

    return nullptr;
  }

  /**
   * Iterates through every token and the corresponding source column numbers
   * and applies a function. We do this to reduce code duplication between
   * buildSegment() and explainIndex()
   */
  template <typename TOKEN_PROCESSOR_T>
  std::exception_ptr forEachOutputToken(
      const std::vector<std::string_view>& input_row,
      TOKEN_PROCESSOR_T token_processor) {
    static_assert(std::is_convertible<TOKEN_PROCESSOR_T,
                                      std::function<void(Token)>>::value);
    std::unordered_map<uint32_t, uint32_t> unigram_to_col_num;
    std::vector<uint32_t> unigram_hashes;
    for (uint32_t col = 0;
         col < std::min<size_t>(input_row.size(), _metadata->numColumns());
         col++) {
      std::string str_val(input_row[col]);
      switch (_metadata->colType(col)) {
        case TabularDataType::Numeric: {
          std::exception_ptr err;
          uint32_t unigram = _metadata->getNumericHashValue(col, str_val, err);
          if (err) {
            return err;
          }
          unigram_to_col_num[unigram] = col;
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Categorical: {
          uint32_t unigram = _metadata->getStringHashValue(str_val, col);
          unigram_to_col_num[unigram] = col;
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Label:
        case TabularDataType::Ignore:
          break;
      }
    }

    std::vector<uint32_t> hashes;
    if (_with_pairgrams) {
      TextEncodingUtils::forEachPairgramFromUnigram(
          unigram_hashes, _output_range,
          [&](TextEncodingUtils::PairGram pairgram) {
            Token token;
            token.token = pairgram.pairgram;
            token.first_column = unigram_to_col_num[pairgram.first_token];
            token.second_column = unigram_to_col_num[pairgram.second_token];
            token_processor(token);
          });
    } else {
      for (auto unigram : unigram_hashes) {
        Token token;
        token.token = unigram % _output_range;
        token.first_column = unigram_to_col_num[unigram];
        token.second_column = token.first_column;
        token_processor(token);
      }
    }
    return nullptr;
  }

  std::pair<uint32_t, uint32_t> getColumnNumbersForIndex(
      const std::vector<std::string_view>& input_row, uint32_t index) {
    std::pair<uint32_t, uint32_t> col_nums;
    if (auto e = forEachOutputToken(input_row, [&](Token token) {
          if (token.token == index) {
            col_nums = {token.first_column, token.second_column};
          }
        })) {
      std::rethrow_exception(e);
    }
    return col_nums;
  }

 private:
  // Private constructor for cereal
  TabularHashFeatures() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _metadata, _output_range,
            _with_pairgrams);
  }

  TabularMetadataPtr _metadata;
  uint32_t _output_range;
  bool _with_pairgrams;
};

using TabularHashFeaturesPtr = std::shared_ptr<TabularHashFeatures>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TabularHashFeatures)
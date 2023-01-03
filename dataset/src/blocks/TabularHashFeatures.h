#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <hashing/src/UniversalHash.h>
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::dataset {

/**
 * @brief Given some metadata about a tabular dataset, assign unique categories
 * to columns and compute either unigrams or pairgrams of the categories
 * depending on the "with_pairgrams" flag.
 */
class TabularHashFeatures final : public Block {
 public:
  TabularHashFeatures(TabularMetadataPtr metadata, uint32_t output_range,
                      bool with_pairgrams = true)
      : _metadata(std::move(metadata)),
        _output_range(output_range),
        _with_pairgrams(with_pairgrams) {}

  struct Token {
    uint32_t token;
    ColumnIdentifier first_column;
    ColumnIdentifier second_column;
  };

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    _metadata->updateColumnNumbers(column_number_map);
  }

  bool hasColumnNames() const final { return _metadata->hasColumnNames(); }

  bool hasColumnNumbers() const final { return _metadata->hasColumnNumbers(); }

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final {
    return _metadata->expectedNumColumnsInRowInput();
  };

  Explanation explainIndex(uint32_t index_within_block,
                           const RowInput& input_row) final {
    return explainIndexImpl(index_within_block, input_row);
  }

  Explanation explainIndex(uint32_t index_within_block,
                           const MapInput& input_map) final {
    return explainIndexImpl(index_within_block, input_map);
  }

  template <typename ColumnarInputType>
  Explanation explainIndexImpl(uint32_t index_within_block,
                               const ColumnarInputType& input) {
    ColumnIdentifier first_column;
    ColumnIdentifier second_column;

    if (auto e = forEachOutputToken(input, [&](Token& token) {
          if (token.token == index_within_block) {
            first_column = std::move(token.first_column);
            second_column = std::move(token.second_column);
          }
        })) {
      std::rethrow_exception(e);
    }

    if (first_column == second_column) {
      return {first_column.name(), std::string(input.at(first_column))};
    }

    auto column_name = first_column.name() + "," + second_column.name();
    auto keyword = std::string(input.at(first_column)) + "," +
                   std::string(input.at(second_column));

    return {column_name, keyword};
  }

 protected:
  std::exception_ptr buildSegment(const RowInput& input_row,
                                  SegmentedFeatureVector& vec) final {
    return buildSegmentImpl(input_row, vec);
  }

  std::exception_ptr buildSegment(const MapInput& input_map,
                                  SegmentedFeatureVector& vec) final {
    return buildSegmentImpl(input_map, vec);
  }

  // TODO(david) We should always include all unigrams but if the number of
  // columns is too large, this processing time becomes slow. One idea is to
  // cap the number of pairgrams at a certain threshold by selecting random
  // pairs of columns to pairgram together.
  template <typename ColumnarInputType>
  std::exception_ptr buildSegmentImpl(const ColumnarInputType& input,
                                      SegmentedFeatureVector& vec) {
    std::unordered_map<uint32_t, uint32_t> unigram_to_col_num;
    std::vector<uint32_t> tokens;
    if (auto e = forEachOutputToken(input, [&](const Token& token) {
          tokens.push_back(token.token);
        })) {
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
  template <typename ColumnarInputType, typename TOKEN_PROCESSOR_T>
  std::exception_ptr forEachOutputToken(const ColumnarInputType& input,
                                        TOKEN_PROCESSOR_T token_processor) {
    static_assert(std::is_convertible<TOKEN_PROCESSOR_T,
                                      std::function<void(Token&)>>::value);
    std::unordered_map<uint32_t, ColumnIdentifier> unigram_to_column_identifier;
    std::vector<uint32_t> unigram_hashes;
    for (uint32_t col = 0; col < _metadata->numColumns(); col++) {
      auto column_identifier = _metadata->columnIdentifier(col);

      std::string str_val(getColumn(input, column_identifier));
      if (str_val.empty()) {
        continue;
      }

      switch (_metadata->colType(col)) {
        case TabularDataType::Numeric: {
          std::exception_ptr err;
          uint32_t unigram = _metadata->getNumericHashValue(col, str_val, err);
          if (err) {
            return err;
          }
          unigram_to_column_identifier[unigram] = std::move(column_identifier);
          unigram_hashes.push_back(unigram);
          break;
        }
        case TabularDataType::Categorical: {
          uint32_t unigram = _metadata->getStringHashValue(str_val, col);
          unigram_to_column_identifier[unigram] = std::move(column_identifier);
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
            token.first_column =
                unigram_to_column_identifier[pairgram.first_token];
            token.second_column =
                unigram_to_column_identifier[pairgram.second_token];
            token_processor(token);
          });
    } else {
      for (auto unigram : unigram_hashes) {
        Token token;
        token.token = unigram % _output_range;
        token.first_column = unigram_to_column_identifier[unigram];
        token.second_column = token.first_column;
        token_processor(token);
      }
    }
    return nullptr;
  }

  std::pair<ColumnIdentifier, ColumnIdentifier> getColumnIdentifiersForIndex(
      const std::vector<std::string_view>& input_row, uint32_t index) {
    std::pair<uint32_t, uint32_t> column_identifiers;
    if (auto e = forEachOutputToken(input_row, [&](Token& token) {
          if (token.token == index) {
            column_identifiers = {std::move(token.first_column),
                                  std::move(token.second_column)};
          }
        })) {
      std::rethrow_exception(e);
    }
    return column_identifiers;
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
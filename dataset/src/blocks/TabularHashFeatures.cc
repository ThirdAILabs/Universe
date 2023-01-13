#include <hashing/src/HashUtils.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::dataset {

Explanation TabularHashFeatures::explainIndex(uint32_t index_within_block,
                                              ColumnarInputSample& input) {
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
    return {first_column.name(), std::string(input.column(first_column))};
  }

  auto column_name = first_column.name() + "," + second_column.name();
  auto keyword = std::string(input.column(first_column)) + "," +
                 std::string(input.column(second_column));

  return {column_name, keyword};
}

std::exception_ptr TabularHashFeatures::buildSegment(
    ColumnarInputSample& input, SegmentedFeatureVector& vec) {
  std::vector<uint32_t> tokens;
  if (auto e = forEachOutputToken(
          input, [&](const Token& token) { tokens.push_back(token.token); })) {
    return e;
  };

  TextEncoding::sumRepeatedIndices(
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
std::exception_ptr TabularHashFeatures::forEachOutputToken(
    ColumnarInputSample& input, TOKEN_PROCESSOR_T token_processor) {
  static_assert(std::is_convertible<TOKEN_PROCESSOR_T,
                                    std::function<void(Token&)>>::value);
  UnigramToColumnIdentifier unigram_to_column_identifier;
  std::vector<uint32_t> unigram_hashes;

  uint32_t salt = 0;
  for (const auto& column : _columns) {
    auto column_identifier = column.identifier;

    std::string str_val(input.column(column_identifier));
    if (str_val.empty()) {
      continue;
    }

    uint32_t unigram;

    switch (column.type) {
      case TabularDataType::Numeric: {
        unigram = computeBin(column, str_val);
        break;
      }
      case TabularDataType::Categorical: {
        unigram = TextEncoding::computeUnigram(str_val.data(), str_val.size());
        break;
      }
    }
    // Hash with different salt per column.
    unigram = hashing::HashUtils::combineHashes(unigram, salt++);

    unigram_to_column_identifier[unigram] = std::move(column_identifier);
    unigram_hashes.push_back(unigram);
  }

  std::vector<uint32_t> hashes;
  if (_with_pairgrams) {
    TextEncoding::forEachPairgramFromUnigram(
        unigram_hashes, _output_range, [&](TextEncoding::PairGram pairgram) {
          auto token =
              Token::fromPairgram(pairgram, unigram_to_column_identifier);
          token_processor(token);
        });
  } else {
    for (auto unigram : unigram_hashes) {
      auto token = Token::fromUnigram(unigram % _output_range,
                                      unigram_to_column_identifier);
      token_processor(token);
    }
  }
  return nullptr;
}

std::vector<ColumnIdentifier*>
TabularHashFeatures::concreteBlockColumnIdentifiers() {
  std::vector<ColumnIdentifier*> identifier_ptrs;
  identifier_ptrs.reserve(_columns.size());
  for (auto& column : _columns) {
    identifier_ptrs.push_back(&column.identifier);
  }
  return identifier_ptrs;
}

uint32_t TabularHashFeatures::computeBin(const TabularColumn& column,
                                         std::string_view str_val) {
  if (str_val.empty()) {
    return column.num_bins.value();
  }
  double value;
  char* end;
  value = std::strtod(str_val.data(), &end);
  if (std::isnan(value)) {
    value = 0.0;
  }

  if (value < column.range->first) {
    return 0;
  }
  if (value > column.range->second) {
    return column.num_bins.value() - 1;
  }

  double binsize = column.binSize();
  if (binsize == 0) {
    return 0;
  }
  return static_cast<uint32_t>(
      std::round((value - column.range->first) / binsize));
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TabularHashFeatures)

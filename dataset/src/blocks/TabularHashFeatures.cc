#include <cereal/archives/binary.hpp>
#include <hashing/src/HashUtils.h>
#include <dataset/src/blocks/TabularHashFeatures.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/Random.h>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <random>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

using UnigramToColumnIdentifier =
    std::unordered_map<uint32_t, ColumnIdentifier>;

struct PairGram {
  uint32_t pairgram;
  uint32_t first_token;
  uint32_t second_token;
};

struct Token {
  static Token fromUnigram(
      uint32_t unigram, const UnigramToColumnIdentifier& to_column_identifier) {
    Token token;
    token.token = unigram;
    token.first_column = to_column_identifier.at(unigram);
    token.second_column = token.first_column;
    return token;
  }

  static Token fromPairgram(
      PairGram pairgram,
      const UnigramToColumnIdentifier& to_column_identifier) {
    Token token;
    token.token = pairgram.pairgram;
    token.first_column = to_column_identifier.at(pairgram.first_token);
    token.second_column = to_column_identifier.at(pairgram.second_token);
    return token;
  }

  uint32_t token;
  ColumnIdentifier first_column;
  ColumnIdentifier second_column;
};

TabularHashFeatures::TabularHashFeatures(
    const std::vector<TabularColumn>& columns, uint32_t output_range,
    bool with_pairgrams)
    : _output_range(output_range), _with_pairgrams(with_pairgrams) {
  std::mt19937 rng(2948);

  // we precompute a random salt value for each column so when we call
  // combineHashes with those values we don't bias the output distribution to
  // have more higher order bits set to zero
  for (const auto& column : columns) {
    uint32_t salt = rng();
    _columns.push_back(std::make_pair(column, salt));
  }
}

Explanation TabularHashFeatures::explainIndex(uint32_t index_within_block,
                                              ColumnarInputSample& input) {
  ColumnIdentifier first_column;
  ColumnIdentifier second_column;

  forEachOutputToken(input, [&](Token& token) {
    if (token.token == index_within_block) {
      first_column = std::move(token.first_column);
      second_column = std::move(token.second_column);
    }
  });

  if (first_column == second_column) {
    return {first_column.name(), input.column(first_column)};
  }

  auto column_name = first_column.name() + "," + second_column.name();
  auto keyword = input.column(first_column) + "," + input.column(second_column);

  return {column_name, keyword};
}

void TabularHashFeatures::buildSegment(ColumnarInputSample& input,
                                       SegmentedFeatureVector& vec) {
  std::vector<uint32_t> tokens;
  forEachOutputToken(
      input, [&tokens](const Token& token) { tokens.push_back(token.token); });

  for (auto& [index, value] : token_encoding::sumRepeatedIndices(tokens)) {
    vec.addSparseFeatureToSegment(index, value);
  }
}

/**
 * Iterates through every token and the corresponding source column numbers
 * and applies a function. We do this to reduce code duplication between
 * buildSegment() and explainIndex()
 */
template <typename TOKEN_PROCESSOR_T>
void TabularHashFeatures::forEachOutputToken(
    ColumnarInputSample& input, TOKEN_PROCESSOR_T token_processor) {
  static_assert(std::is_convertible<TOKEN_PROCESSOR_T,
                                    std::function<void(Token&)>>::value);
  UnigramToColumnIdentifier unigram_to_column_identifier;
  std::vector<uint32_t> unigram_hashes;

  for (const auto& [column, salt] : _columns) {
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
        unigram =
            token_encoding::seededMurmurHash(str_val.data(), str_val.size());
        break;
      }
      default: {
        throw std::invalid_argument(
            "Column type was expected to be numeric or categorical.");
      }
    }
    // Hash with different salt per column so the same bin in a different
    // column doesn't map to the same unigram.
    unigram = hashing::combineHashes(unigram, salt);

    unigram_to_column_identifier[unigram] = std::move(column_identifier);
    unigram_hashes.push_back(unigram);
  }

  std::vector<uint32_t> hashes;
  if (_with_pairgrams) {
    forEachPairgramFromUnigram(
        unigram_hashes, _output_range, [&](PairGram pairgram) {
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
}

std::vector<ColumnIdentifier*>
TabularHashFeatures::concreteBlockColumnIdentifiers() {
  std::vector<ColumnIdentifier*> identifier_ptrs;
  identifier_ptrs.reserve(_columns.size());
  for (auto& [column, _] : _columns) {
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
  if (value >= column.range->second) {
    return column.num_bins.value() - 1;
  }

  double binsize = column.binSize();
  if (binsize == 0) {
    return 0;
  }
  return static_cast<uint32_t>(
      std::round((value - column.range->first) / binsize));
}

template <typename PAIRGRAM_PROCESSOR_T>
void forEachPairgramFromUnigram(const std::vector<uint32_t>& unigram_hashes,
                                uint32_t output_range,
                                PAIRGRAM_PROCESSOR_T pairgram_processor) {
  static_assert(std::is_convertible<PAIRGRAM_PROCESSOR_T,
                                    std::function<void(PairGram)>>::value);

  for (uint32_t token = 0; token < unigram_hashes.size(); token++) {
    for (uint32_t prev_token = 0; prev_token <= token; prev_token++) {
      uint32_t combined_hash = hashing::combineHashes(
          unigram_hashes[prev_token], unigram_hashes[token]);
      combined_hash = combined_hash % output_range;
      pairgram_processor({/* pairgram= */ combined_hash,
                          /* first_token= */ unigram_hashes[prev_token],
                          /* second_token= */ unigram_hashes[token]});
    }
  }
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TabularHashFeatures)

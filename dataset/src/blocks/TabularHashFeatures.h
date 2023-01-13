#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <hashing/src/UniversalHash.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

static constexpr uint32_t DEFAULT_NUM_BINS = 10;

enum class TabularDataType {
  Numeric,
  Categorical,
};  // TODO(david) add datetime/text support

struct TabularColumn {
  static TabularColumn Numeric(ColumnIdentifier identifier,
                               std::pair<double, double> range,
                               uint32_t num_bins = DEFAULT_NUM_BINS) {
    TabularColumn tabular_column;
    tabular_column.identifier = std::move(identifier);
    tabular_column.type = TabularDataType::Numeric;
    tabular_column.range = range;
    tabular_column.num_bins = num_bins;
    return tabular_column;
  }

  static TabularColumn Categorical(ColumnIdentifier identifier) {
    TabularColumn tabular_column;
    tabular_column.identifier = std::move(identifier);
    tabular_column.type = TabularDataType::Categorical;
    return tabular_column;
  }

  double binSize() const { return (range->second - range->first) / *num_bins; }

  ColumnIdentifier identifier;
  TabularDataType type;
  std::optional<std::pair<double, double>> range = std::nullopt;
  std::optional<uint32_t> num_bins = std::nullopt;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(identifier, type, range, num_bins);
  }
};

/**
 * @brief Given some metadata about a tabular dataset, assign unique categories
 * to columns and compute either unigrams or pairgrams of the categories
 * depending on the "with_pairgrams" flag.
 */
class TabularHashFeatures final : public Block {
 public:
  TabularHashFeatures(std::vector<TabularColumn> columns, uint32_t output_range,
                      bool with_pairgrams = true)
      : _columns(std::move(columns)),
        _output_range(output_range),
        _with_pairgrams(with_pairgrams) {}

  using UnigramToColumnIdentifier =
      std::unordered_map<uint32_t, ColumnIdentifier>;

  struct Token {
    static Token fromUnigram(
        uint32_t unigram,
        const UnigramToColumnIdentifier& to_column_identifier) {
      Token token;
      token.token = unigram;
      token.first_column = to_column_identifier.at(unigram);
      token.second_column = token.first_column;
      return token;
    }

    static Token fromPairgram(
        TokenEncoding::PairGram pairgram,
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

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

 protected:
  std::exception_ptr buildSegment(ColumnarInputSample& input,
                                  SegmentedFeatureVector& vec) final;

  /**
   * Iterates through every token and the corresponding source column numbers
   * and applies a function. We do this to reduce code duplication between
   * buildSegment() and explainIndex()
   */
  template <typename TOKEN_PROCESSOR_T>
  std::exception_ptr forEachOutputToken(ColumnarInputSample& input,
                                        TOKEN_PROCESSOR_T token_processor);

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final;

 private:
  /**
   * For values between the min and the max (inclusive), we divide that range
   * into N uniform chunks and return a bin number from 0 to N-1. Additionally
   * we support three special cases of bins:
   *    - if we're given an empty str_val we return bin number N
   *    - if the str_val is less than the min we return bin number 0
   *    - if the str_val is greater than the max we return bin number N - 1
   */
  static uint32_t computeBin(const TabularColumn& column,
                             std::string_view str_val);

  // Private constructor for cereal
  TabularHashFeatures() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _columns, _output_range,
            _with_pairgrams);
  }

  std::vector<TabularColumn> _columns;
  uint32_t _output_range;
  bool _with_pairgrams;
};

using TabularHashFeaturesPtr = std::shared_ptr<TabularHashFeatures>;

}  // namespace thirdai::dataset
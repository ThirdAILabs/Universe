#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include "BlockInterface.h"
#include <hashing/src/UniversalHash.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::dataset {

static constexpr uint32_t DEFAULT_NUM_BINS = 10;

enum class TabularDataType {
  Numeric,
  Categorical,
};

struct TabularColumn {
  static TabularColumn Numeric(ColumnIdentifier identifier,
                               std::pair<double, double> range,
                               uint32_t num_bins = DEFAULT_NUM_BINS) {
    TabularColumn tabular_column;
    tabular_column.identifier = std::move(identifier);
    tabular_column.type = TabularDataType::Numeric;
    tabular_column.range = range;
    tabular_column.num_bins = num_bins;
    tabular_column.bin_size = (range.second - range.first) / num_bins;
    return tabular_column;
  }

  static TabularColumn Categorical(ColumnIdentifier identifier) {
    TabularColumn tabular_column;
    tabular_column.identifier = std::move(identifier);
    tabular_column.type = TabularDataType::Categorical;
    return tabular_column;
  }

  double binSize() const { return *bin_size; }

  ColumnIdentifier identifier;
  TabularDataType type;
  std::optional<std::pair<double, double>> range = std::nullopt;
  std::optional<uint32_t> num_bins = std::nullopt;
  std::optional<double> bin_size = std::nullopt;

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(identifier, type, range, num_bins, bin_size);
  }
};

/**
 * @brief Given some metadata about a tabular dataset, assign unique categories
 * to columns and compute either unigrams or pairgrams of the categories
 * depending on the "with_pairgrams" flag.
 */
class TabularHashFeatures final : public Block {
 public:
  TabularHashFeatures(const std::vector<TabularColumn>& columns,
                      uint32_t output_range, bool with_pairgrams = true);

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

 protected:
  void buildSegment(ColumnarInputSample& input,
                    SegmentedFeatureVector& vec) final;

  /**
   * Iterates through every token and the corresponding source column numbers
   * and applies a function. We do this to reduce code duplication between
   * buildSegment() and explainIndex()
   */
  template <typename TOKEN_PROCESSOR_T>
  void forEachOutputToken(ColumnarInputSample& input,
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

  std::vector<std::pair<TabularColumn, uint32_t>> _columns;
  uint32_t _output_range;
  bool _with_pairgrams;
};

using TabularHashFeaturesPtr = std::shared_ptr<TabularHashFeatures>;

}  // namespace thirdai::dataset
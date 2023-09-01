#pragma once

#include <hashing/src/HashUtils.h>
#include <data/src/transformations/Transformation.h>
#include <memory>
#include <utility>

namespace thirdai::data {

/**
 * This transformation is intended to be used to aggregate multiple columns into
 * a single indices column and and a single values column. This pair of indices,
 * values can then be converted into a dataset to be passed to bolt. This is so
 * that the logic to convert ColumnMaps to dataset can be specific to a single
 * pair of columns instead of needing to take in a list of columns. It also
 * seperates the logic so that we could add additional mechanisms for
 * aggregating multiple columns, for instance concatentation. The aggregation is
 * done by feature hashing.
 */
class FeatureHash final : public Transformation {
 public:
  FeatureHash(std::vector<std::string> input_columns,
              std::string output_indices_column,
              std::string output_values_columns, size_t hash_range);

  static std::shared_ptr<FeatureHash> make(
      std::vector<std::string> input_columns, std::string output_indices_column,
      std::string output_values_columns, size_t hash_range) {
    return std::make_shared<FeatureHash>(
        std::move(input_columns), std::move(output_indices_column),
        std::move(output_values_columns), hash_range);
  }

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  const auto& inputColumns() const { return _input_columns; }

 private:
  inline uint32_t hash(uint32_t index, uint32_t column_salt) const {
    return hashing::combineHashes(index, column_salt) % _hash_range;
  }

  static uint32_t columnSalt(const std::string& name) {
    return hashing::MurmurHash(name.data(), name.size(), 932042);
  }

  size_t _hash_range;

  std::vector<std::string> _input_columns;
  std::string _output_indices_column;
  std::string _output_values_column;

  FeatureHash() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data
#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class DyadicInterval final : public Transformation {
 public:
  DyadicInterval(std::string input_column, std::string output_interval_prefix,
                 std::string target_column, size_t n_intervals);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ColumnMap inferenceFeaturization(ColumnMap columns) const;

 private:
  static std::vector<size_t> computeOffsets(
      const ArrayColumnBasePtr<uint32_t>& texts, size_t chunk_size);

  std::string _input_column;
  std::string _output_interval_prefix;
  std::string _target_column;

  size_t _n_intervals;

  DyadicInterval() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data
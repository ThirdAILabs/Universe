#pragma once

#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>

namespace thirdai::dataset {

// Hashes a string columnm into a given output range using MurmurHash.
class StringHash final : public Transformation {
 public:
  StringHash(std::string input_column_name, std::string output_column_name,
             uint32_t output_range, uint32_t seed = time(nullptr))
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range),
        _seed(seed) {}

  void apply(ColumnMap& columns) final;

 private:
  uint32_t hash(const std::string& str) const;

  std::string _input_column_name;
  std::string _output_column_name;
  uint32_t _output_range;
  uint32_t _seed;
};

}  // namespace thirdai::dataset
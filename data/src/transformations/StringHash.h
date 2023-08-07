#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

// Hashes a string columnm into a given output range using MurmurHash. Uses a
// default seed so our data pipeline is reproducible.
class StringHash final : public Transformation {
 public:
  StringHash(std::string input_column_name, std::string output_column_name,
             std::optional<uint32_t> output_range = std::nullopt,
             uint32_t seed = 42)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range),
        _seed(seed) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

  proto::data::Transformation* toProto() const final;

 private:
  uint32_t hash(const std::string& str) const;

  std::string _input_column_name;
  std::string _output_column_name;
  std::optional<uint32_t> _output_range;
  uint32_t _seed;
};

}  // namespace thirdai::data

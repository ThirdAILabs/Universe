#pragma once

#include <data/src/transformations/Transformation.h>
#include <memory>
#include <optional>

namespace thirdai::data {

// Hashes a string columnm into a given output range using MurmurHash. Uses a
// default seed so our data pipeline is reproducible.
class StringHash final : public Transformation {
 public:
  StringHash(std::string input_column_name, std::string output_column_name,
             std::optional<uint32_t> output_range = std::nullopt,
             std::optional<char> delimiter = std::nullopt, uint32_t seed = 42)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _output_range(output_range),
        _delimiter(delimiter),
        _seed(seed) {}

  static std::shared_ptr<StringHash> make(
      std::string input_column_name, std::string output_column_name,
      std::optional<uint32_t> output_range = std::nullopt,
      std::optional<char> delimiter = std::nullopt, uint32_t seed = 42) {
    return std::make_shared<StringHash>(std::move(input_column_name),
                                        std::move(output_column_name),
                                        output_range, delimiter, seed);
  }

  explicit StringHash(const proto::data::StringHash& string_hash);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

 private:
  uint32_t hash(const std::string& str) const;

  std::string _input_column_name;
  std::string _output_column_name;
  std::optional<uint32_t> _output_range;
  std::optional<char> _delimiter;
  uint32_t _seed;
};

}  // namespace thirdai::data

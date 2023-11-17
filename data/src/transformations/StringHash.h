#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <data/src/transformations/Transformation.h>
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

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "string_hash"; }

 private:
  // Private constructor for cereal.
  StringHash()
      : _input_column_name(),
        _output_column_name(),
        _output_range(0),
        _seed(0) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Transformation>(this), _input_column_name,
            _output_column_name, _output_range, _delimiter, _seed);
  }

  uint32_t hash(const std::string& str) const;

  std::string _input_column_name;
  std::string _output_column_name;
  std::optional<uint32_t> _output_range;
  std::optional<char> _delimiter;
  uint32_t _seed;
};

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::StringHash)

#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>

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

  void apply(ColumnMap& columns) final;

  void backpropagate(ContributionColumnMap& contribuition_columns) final;

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
            _output_column_name, _output_range, _seed);
  }

  uint32_t hash(const std::string& str) const;

  std::string _input_column_name;
  std::string _output_column_name;
  std::optional<uint32_t> _output_range;
  uint32_t _seed;
};

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::StringHash)

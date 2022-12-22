#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::data {

// Bins a dense float column into categorical sparse values. If the input column
// is the same as the output column then that column will be replaced in the
// column map.
class BinningTransformation final : public Transformation {
 public:
  BinningTransformation(std::string input_column_name,
                        std::string output_column_name,
                        float inclusive_min_value, float exclusive_max_value,
                        uint32_t num_bins)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _inclusive_min_value(inclusive_min_value),
        _exclusive_max_value(exclusive_max_value),
        _binsize((exclusive_max_value - inclusive_min_value) / num_bins),
        _num_bins(num_bins) {}

  void apply(ColumnMap& columns, bool prepare_for_backpropagate) final;

  void backpropagate(ColumnMap& columns,
                     ContributionColumnMap& contribuition_columns) final;

 private:
  // Private constructor for cereal.
  BinningTransformation()
      : _input_column_name(),
        _output_column_name(),
        _inclusive_min_value(0),
        _exclusive_max_value(0),
        _binsize(0),
        _num_bins(0) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Transformation>(this), _input_column_name,
            _output_column_name, _inclusive_min_value, _exclusive_max_value,
            _binsize, _num_bins);
  }

  std::optional<uint32_t> getBin(float value) const;

  std::string _input_column_name;
  std::string _output_column_name;

  float _inclusive_min_value;
  float _exclusive_max_value;
  float _binsize;
  uint32_t _num_bins;
};

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::BinningTransformation)
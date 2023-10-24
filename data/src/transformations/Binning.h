#pragma once

#include <data/src/transformations/Transformation.h>
#include <exception>
#include <memory>
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

  static std::shared_ptr<BinningTransformation> make(
      std::string input_column_name, std::string output_column_name,
      float inclusive_min_value, float exclusive_max_value, uint32_t num_bins) {
    return std::make_shared<BinningTransformation>(
        std::move(input_column_name), std::move(output_column_name),
        inclusive_min_value, exclusive_max_value, num_bins);
  }

  explicit BinningTransformation(const proto::data::Binning& binning);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

 private:
  std::optional<uint32_t> getBin(float value) const;

  std::string _input_column_name;
  std::string _output_column_name;

  float _inclusive_min_value;
  float _exclusive_max_value;
  float _binsize;
  uint32_t _num_bins;
};

}  // namespace thirdai::data

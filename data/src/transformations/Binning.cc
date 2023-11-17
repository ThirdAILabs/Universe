#include "Binning.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/columns/ValueColumns.h>
#include <string>

namespace thirdai::data {

ColumnMap BinningTransformation::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto column = columns.getValueColumn<float>(_input_column_name);

  std::vector<uint32_t> binned_values(column->numRows());

  std::optional<float> invalid_value = std::nullopt;
#pragma omp parallel for default(none) \
    shared(column, binned_values, invalid_value) if (columns.numRows() > 1)
  for (uint64_t i = 0; i < column->numRows(); i++) {
    if (auto bin = getBin(column->value(i))) {
      binned_values[i] = *bin;
    } else {
#pragma omp critical
      invalid_value = column->value(i);
    }
  }

  if (invalid_value) {
    throw std::invalid_argument(
        "Cannot bin value " + std::to_string(invalid_value.value()) +
        ". Expected values in range [" + std::to_string(_inclusive_min_value) +
        ", " + std::to_string(_exclusive_max_value) + ").");
  }

  auto output_column =
      ValueColumn<uint32_t>::make(std::move(binned_values), _num_bins);

  columns.setColumn(_output_column_name, output_column);

  return columns;
}

std::optional<uint32_t> BinningTransformation::getBin(float value) const {
  if (value < _inclusive_min_value) {
    return 0;
  }
  if (value >= _exclusive_max_value) {
    return _num_bins - 1;
  }
  return (value - _inclusive_min_value) / _binsize;
}

void BinningTransformation::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  (void)state;

  float value = input.getValueColumn<float>(_input_column_name)->value(0);
  uint32_t bin = getBin(value).value();

  explanations.store(
      _output_column_name, bin,
      explanations.explain(_input_column_name, /* feature_index= */ 0));
}

ar::ConstArchivePtr BinningTransformation::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_column", ar::str(_input_column_name));
  map->set("output_column", ar::str(_output_column_name));
  map->set("min", ar::f32(_inclusive_min_value));
  map->set("max", ar::f32(_exclusive_max_value));
  map->set("binsize", ar::f32(_binsize));
  map->set("num_bins", ar::u64(_num_bins));

  return map;
}

BinningTransformation::BinningTransformation(const ar::Archive& archive)
    : _input_column_name(archive.str("input_column")),
      _output_column_name(archive.str("output_column")),
      _inclusive_min_value(archive.getAs<ar::F32>("min")),
      _exclusive_max_value(archive.getAs<ar::F32>("max")) {}

}  // namespace thirdai::data
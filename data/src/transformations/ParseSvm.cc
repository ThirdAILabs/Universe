#include "ParseSvm.h"
#include <data/src/columns/ArrayColumns.h>
#include <utils/text/StringManipulation.h>
#include <stdexcept>
#include <string>

namespace thirdai::data {

ParseSvm::ParseSvm(std::string input_col, std::string indices_col,
                   std::string values_col, std::string labels_col,
                   size_t indices_dim, size_t labels_dim)
    : _input_column(std::move(input_col)),
      _indices_col(std::move(indices_col)),
      _values_col(std::move(values_col)),
      _labels_col(std::move(labels_col)),
      _indices_dim(indices_dim),
      _labels_dim(labels_dim) {}

ColumnMap ParseSvm::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto input = columns.getValueColumn<std::string>(_input_column);

  std::vector<std::vector<uint32_t>> indices(columns.numRows());
  std::vector<std::vector<float>> values(columns.numRows());
  std::vector<std::vector<uint32_t>> labels(columns.numRows());

  for (size_t i = 0; i < columns.numRows(); i++) {
    auto items = text::splitOnWhiteSpace(input->value(i));

    for (const auto& label : text::split(items[0], ',')) {
      labels[i].push_back(std::stoul(label));
    }

    for (auto kv_pair = items.begin() + 1; kv_pair != items.end(); kv_pair++) {
      auto loc = kv_pair->find(':');

      indices[i].push_back(std::stoul(kv_pair->substr(0, loc)));
      values[i].push_back(std::stof(kv_pair->substr(loc + 1)));
    }
  }

  columns.setColumn(_indices_col, ArrayColumn<uint32_t>::make(
                                      std::move(indices), _indices_dim));
  columns.setColumn(_values_col, ArrayColumn<float>::make(std::move(values)));
  columns.setColumn(
      _labels_col, ArrayColumn<uint32_t>::make(std::move(labels), _labels_dim));

  return columns;
}

ar::ConstArchivePtr ParseSvm::toArchive() const {
  throw std::invalid_argument(
      "toArchive is not implemented for ParseSvm transformation");
}

}  // namespace thirdai::data
#include "SpladeMachAugmentation.h"
#include <data/src/columns/ValueColumns.h>
#include <string>

namespace thirdai::data {

ColumnMap SpladeMachAugmentation::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto input = columns.getValueColumn<std::string>(_input_column);

  std::vector<std::string> output(input->numRows());

  for (size_t i = 0; i < output.size(); i += 4096) {
    std::vector<std::string> batch;
    for (size_t j = i; j < std::min(i + 4096, output.size()); j++) {
      batch.push_back(input->value(j));
    }

    auto hashes = _model->getTopHashBuckets(batch, _n_hashes_per_model);

    for (size_t j = 0; j < hashes.size(); j++) {
      std::string joined;
      for (uint32_t hash : hashes.at(j)) {
        joined += std::to_string(hash) + ' ';
      }
      if (!joined.empty()) {
        joined.pop_back();
      }
      output[i + j] = joined;
    }
  }

  columns.setColumn(_output_column,
                    data::ValueColumn<std::string>::make(std::move(output)));

  return columns;
}

}  // namespace thirdai::data
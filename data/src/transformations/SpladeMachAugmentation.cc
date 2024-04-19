#include "SpladeMachAugmentation.h"
#include <archive/src/Archive.h>
#include <auto_ml/src/pretrained/SpladeMach.h>
#include <data/src/columns/ArrayColumns.h>
#include <string>

namespace thirdai::data {

ColumnMap SpladeMachAugmentation::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto input = columns.getValueColumn<std::string>(_input_column);

  std::vector<std::vector<uint32_t>> output(input->numRows());

  const size_t batch_size = 4096;
  for (size_t start = 0; start < output.size(); start += batch_size) {
    const size_t end = std::min(start + batch_size, output.size());
    std::vector<std::string> batch;
    for (size_t j = start; j < end; j++) {
      batch.push_back(input->value(j));
    }

    auto hashes = _model->getTopHashBuckets(batch, _n_hashes_per_model);

    for (size_t j = 0; j < hashes.size(); j++) {
      output[start + j] = std::move(hashes[j]);
    }
  }

  columns.setColumn(_output_column,
                    data::ArrayColumn<uint32_t>::make(std::move(output),
                                                      _model->outputDim()));

  return columns;
}

ar::ConstArchivePtr SpladeMachAugmentation::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("input_column", ar::str(_input_column));
  map->set("output_column", ar::str(_output_column));

  map->set("model", _model->toArchive());
  map->set("n_hashes_per_model", ar::u64(_n_hashes_per_model));

  return map;
}

SpladeMachAugmentation::SpladeMachAugmentation(const ar::Archive& archive)
    : SpladeMachAugmentation(
          archive.str("input_column"), archive.str("output_column"),
          automl::SpladeMach::fromArchive(*archive.get("model")),
          archive.u64("n_hashes_per_model")) {}

}  // namespace thirdai::data
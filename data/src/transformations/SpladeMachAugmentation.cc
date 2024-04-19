#include "SpladeMachAugmentation.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <auto_ml/src/pretrained/SpladeMach.h>
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
        auto token = hash;
        if(_token_offset){
          token += *_token_offset;
        }
          joined += std::to_string(token) + ' ';
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

ar::ConstArchivePtr SpladeMachAugmentation::toArchive() const {
  auto map = ar::Map::make();

  map->set("input_column", ar::str(_input_column));
  map->set("output_column", ar::str(_output_column));

  map->set("model", _model->toArchive());
  map->set("n_hashes_per_model", ar::u64(_n_hashes_per_model));

  if(_token_offset){
    map->set("token_offset", ar::u64(*_token_offset));
  }

  return map;
}

SpladeMachAugmentation::SpladeMachAugmentation(const ar::Archive& archive)
    : _input_column(archive.str("input_column")),
      _output_column(archive.str("output_column")),
      _model(automl::SpladeMach::fromArchive(*archive.get("model"))),
      _n_hashes_per_model(archive.u64("n_hashes_per_model")), 
      _token_offset(archive.getOpt<ar::U64>("token_offset")) {}

}  // namespace thirdai::data
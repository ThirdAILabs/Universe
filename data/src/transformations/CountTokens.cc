#include "CountTokens.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/columns/ValueColumns.h>

namespace thirdai::data {

thirdai::data::ColumnMap thirdai::data::CountTokens::apply(ColumnMap columns,
                                                           State& state) const {
  (void)state;
  auto tokens_column = columns.getArrayColumn<uint32_t>(_input_column);
  std::vector<uint32_t> num_tokens(tokens_column->numRows());

#pragma omp parallel for default(none) \
    shared(tokens_column, num_tokens) if (columns.numRows() > 1)
  for (uint32_t i = 0; i < tokens_column->numRows(); i++) {
    num_tokens[i] = tokens_column->row(i).size();
    if (_max_tokens && num_tokens[i] > _max_tokens) {
      num_tokens[i] = *_max_tokens;
    }
  }

  std::optional<uint32_t> dim =
      _max_tokens ? std::make_optional(*_max_tokens + 1) : std::nullopt;

  auto new_column = ValueColumn<uint32_t>::make(
      /* data= */ std::move(num_tokens), /* dim= */ dim);
  columns.setColumn(/* name= */ _output_column,
                    /* column= */ new_column);
  return columns;
}

ar::ConstArchivePtr CountTokens::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_column", ar::str(_input_column));
  map->set("output_column", ar::str(_output_column));
  if (_max_tokens) {
    map->set("max_tokens", ar::u64(*_max_tokens));
  }

  return map;
}

CountTokens::CountTokens(const ar::Archive& archive)
    : _input_column(archive.str("input_column")),
      _output_column(archive.str("output_column")),
      _max_tokens(archive.getOpt<ar::U64>("max_tokens")) {}

}  // namespace thirdai::data

#include "CountTokens.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
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

template void CountTokens::serialize(cereal::BinaryInputArchive&);
template void CountTokens::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void CountTokens::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _output_column, _max_tokens);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::CountTokens)

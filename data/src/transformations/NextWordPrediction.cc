#include "NextWordPrediction.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <utility>

namespace thirdai::data {

NextWordPrediction::NextWordPrediction(std::string input_column,
                                       std::string context_column,
                                       std::string target_column)
    : _input_column(std::move(input_column)),
      _context_column(std::move(context_column)),
      _target_column(std::move(target_column)) {}

std::vector<size_t> computeOffsets(const ArrayColumnBasePtr<uint32_t>& texts) {
  std::vector<size_t> offsets(texts->numRows() + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < texts->numRows(); i++) {
    offsets[i + 1] = offsets[i] + texts->row(i).size() - 1;
  }
  return offsets;
}

ColumnMap NextWordPrediction::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto texts = columns.getArrayColumn<uint32_t>(_input_column);
  auto sample_offsets = computeOffsets(texts);

  std::vector<std::vector<uint32_t>> contexts;
  contexts.assign(sample_offsets.back(), {});
  std::vector<uint32_t> targets(sample_offsets.back());

  std::exception_ptr error;

#pragma omp parallel for default(none) \
    shared(texts, contexts, targets, sample_offsets, error)
  for (size_t i = 0; i < texts->numRows(); i += 1) {
    try {
      auto tokens = texts->row(i);

      size_t sample_offset = sample_offsets[i];
      for (size_t start = 1; start < tokens.size(); start += 1) {
        contexts[sample_offset] = tokens.range(0, start);
        targets[sample_offset] = tokens[start];
        sample_offset += 1;
      }

    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }
  if (error) {
    std::rethrow_exception(error);
  }
  std::unordered_map<std::string, ColumnPtr> output_columns;
  output_columns[_context_column] =
      ArrayColumn<uint32_t>::make(std::move(contexts), texts->dim());
  output_columns[_target_column] =
      ValueColumn<uint32_t>::make(std::move(targets), texts->dim());

  return ColumnMap(output_columns);
}

ar::ConstArchivePtr NextWordPrediction::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("input_column", ar::str(_input_column));
  map->set("context_column", ar::str(_context_column));
  map->set("target_column", ar::str(_target_column));

  return map;
}

NextWordPrediction::NextWordPrediction(const ar::Archive& archive)
    : _input_column(archive.str("input_column")),
      _context_column(archive.str("context_column")),
      _target_column(archive.str("target_column")) {}
}  // namespace thirdai::data
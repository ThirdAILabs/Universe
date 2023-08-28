#include "MachLabel.h"
#include <data/src/columns/ArrayColumns.h>
#include <exception>

namespace thirdai::data {

MachLabel::MachLabel(std::string input_column_name,
                     std::string output_column_name)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)) {}

MachLabel::MachLabel(const proto::data::MachLabel& mach_label)
    : _input_column_name(mach_label.input_column()),
      _output_column_name(mach_label.output_column()) {}

ColumnMap MachLabel::apply(ColumnMap columns, State& state) const {
  auto entities_column = columns.getArrayColumn<uint32_t>(_input_column_name);

  std::vector<std::vector<uint32_t>> hashes(entities_column->numRows());

  const auto& index = state.machIndex();

  std::exception_ptr error;

#pragma omp parallel for default(none) \
    shared(entities_column, hashes, index, error) if (columns.numRows() > 1)
  for (size_t i = 0; i < entities_column->numRows(); i++) {
    for (uint32_t entity : entities_column->row(i)) {
      try {
        auto entity_hashes = index->getHashes(entity);

        hashes[i].insert(hashes[i].end(), entity_hashes.begin(),
                         entity_hashes.end());

      } catch (...) {
#pragma omp critical
        error = std::current_exception();
      }
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  auto output_column =
      ArrayColumn<uint32_t>::make(std::move(hashes), index->numBuckets());
  columns.setColumn(_output_column_name, output_column);

  return columns;
}

proto::data::Transformation* MachLabel::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* mach_label = transformation->mutable_mach_label();

  mach_label->set_input_column(_input_column_name);
  mach_label->set_output_column(_output_column_name);

  return transformation;
}

}  // namespace thirdai::data

#include "TensorConversion.h"
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <exception>
#include <stdexcept>

namespace thirdai::data {

std::vector<bolt::TensorList> toTensorBatches(
    const ColumnMap& columns, const OutputColumnsList& columns_to_convert,
    size_t batch_size) {
  size_t num_batches = (columns.numRows() + batch_size - 1) / batch_size;
  std::vector<bolt::TensorList> tensors(num_batches);

  for (const auto& column_info : columns_to_convert) {
    auto indices = columns.getArrayColumn<uint32_t>(column_info.indices());

    if (!indices->dim()) {
      throw std::invalid_argument(
          "No dimension found for column '" + column_info.indices() +
          "'. Indices must have dimension to convert to tensor.");
    }

    ArrayColumnBasePtr<float> values = nullptr;
    ValueFillType value_fill_type = ValueFillType::Ones;
    if (column_info.values() && columns.containsColumn(*column_info.values())) {
      values = columns.getArrayColumn<float>(*column_info.values());
    } else {
      if (column_info.valueFillType() == ValueFillType::None) {
        throw std::invalid_argument(
            "Value column was not present in ColumnMap, and no fallback fill "
            "type was specified.");
      }
      value_fill_type = column_info.valueFillType();
    }

    std::exception_ptr error;

#pragma omp parallel for default(none)                                 \
    shared(num_batches, batch_size, columns, indices, values, tensors, \
           value_fill_type, error) if (num_batches > 1)
    for (size_t batch = 0; batch < num_batches; batch++) {
      size_t batch_start = batch * batch_size;
      size_t batch_end = std::min((batch + 1) * batch_size, columns.numRows());

      size_t batch_nonzeros = 0;
      for (size_t i = batch_start; i < batch_end; i++) {
        batch_nonzeros += indices->row(i).size();
      }

      std::vector<uint32_t> batch_indices;
      batch_indices.reserve(batch_nonzeros);
      std::vector<float> batch_values;
      batch_values.reserve(batch_nonzeros);
      std::vector<size_t> batch_lens;
      batch_lens.reserve(batch_end - batch_start);

      for (size_t i = batch_start; i < batch_end; i++) {
        auto indices_row = indices->row(i);

        // Values are optional for converting sparse data. If not specified
        // values are assumed to be 1.0.
        if (values) {
          auto values_row = values->row(i);
          if (indices_row.size() != values_row.size()) {
#pragma omp critical
            error = std::make_exception_ptr(std::invalid_argument(
                "Indices size does not batch values size in row " +
                std::to_string(i) + "."));
            break;
          }
          batch_values.insert(batch_values.end(), values_row.begin(),
                              values_row.end());
        } else {
          float fill_value = (value_fill_type == ValueFillType::SumToOne)
                                 ? values->row(i)[0] / indices_row.size()
                                 : 1.0;

          for (size_t j = 0; j < indices_row.size(); j++) {
            batch_values.push_back(fill_value);
          }
        }

        batch_indices.insert(batch_indices.end(), indices_row.begin(),
                             indices_row.end());
        batch_lens.push_back(indices_row.size());
      }

      tensors[batch].emplace_back(bolt::Tensor::sparse(
          std::move(batch_indices), std::move(batch_values),
          std::move(batch_lens), indices->dim().value()));
    }

    if (error) {
      std::rethrow_exception(error);
    }
  }

  return tensors;
}

bolt::TensorList toTensors(const ColumnMap& columns,
                           const OutputColumnsList& columns_to_convert) {
  return toTensorBatches(columns, columns_to_convert,
                         /* batch_size= */ columns.numRows())
      .at(0);
}

}  // namespace thirdai::data
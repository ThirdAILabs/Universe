#include "SmxTensorConversion.h"
#include <smx/src/tensor/CsrTensor.h>

namespace thirdai::data {

SmxDataset toSmxTensorBatches(const ColumnMap& columns,
                              const OutputColumnsList& columns_to_convert,
                              size_t batch_size) {
  size_t num_batches = (columns.numRows() + batch_size - 1) / batch_size;
  std::vector<std::vector<smx::TensorPtr>> tensors(num_batches);

  for (const auto& column_info : columns_to_convert) {
    if (auto value_col = ValueColumnBase<uint32_t>::cast(
            columns.getColumn(column_info.indices()))) {
      for (size_t batch = 0; batch < num_batches; batch++) {
        size_t batch_start = batch * batch_size;
        size_t batch_end =
            std::min((batch + 1) * batch_size, columns.numRows());

        std::vector<uint32_t> indices;
        indices.reserve(batch_end - batch_start);
        for (size_t i = batch_start; i < batch_end; i++) {
          indices.push_back(value_col->value(i));
        }

        tensors[batch].push_back(
            smx::DenseTensor::make(indices, smx::Shape(indices.size())));
      }
    } else {
      auto indices = columns.getArrayColumn<uint32_t>(column_info.indices());

      if (!indices->dim()) {
        throw std::invalid_argument(
            "No dimension found for column '" + column_info.indices() +
            "'. Indices must have dimension to convert to tensor.");
      }

      ArrayColumnBasePtr<float> values = nullptr;
      ValueFillType value_fill_type = ValueFillType::Ones;
      if (column_info.values()) {
        values = columns.getArrayColumn<float>(*column_info.values());
      } else {
        value_fill_type = column_info.valueFillType();
      }

      std::exception_ptr error;

#pragma omp parallel for default(none)                                 \
    shared(num_batches, batch_size, columns, indices, values, tensors, \
           value_fill_type, error)
      for (size_t batch = 0; batch < num_batches; batch++) {
        size_t batch_start = batch * batch_size;
        size_t batch_end =
            std::min((batch + 1) * batch_size, columns.numRows());

        size_t batch_nonzeros = 0;
        for (size_t i = batch_start; i < batch_end; i++) {
          batch_nonzeros += indices->row(i).size();
        }

        std::vector<uint32_t> batch_indices;
        batch_indices.reserve(batch_nonzeros);
        std::vector<float> batch_values;
        batch_values.reserve(batch_nonzeros);
        std::vector<uint32_t> batch_offsets;
        batch_offsets.reserve(batch_end - batch_start + 1);
        batch_offsets.push_back(0);

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
                                   ? 1.0 / indices_row.size()
                                   : 1.0;

            for (size_t j = 0; j < indices_row.size(); j++) {
              batch_values.push_back(fill_value);
            }
          }

          batch_indices.insert(batch_indices.end(), indices_row.begin(),
                               indices_row.end());
          batch_offsets.push_back(batch_indices.size());
        }

        tensors[batch].push_back(smx::CsrTensor::make(
            batch_offsets, batch_indices, batch_values,
            smx::Shape(batch_end - batch_start, *indices->dim())));
      }

      if (error) {
        std::rethrow_exception(error);
      }
    }
  }

  return tensors;
}

}  // namespace thirdai::data
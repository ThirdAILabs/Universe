#include "TensorConversion.h"
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <exception>
#include <stdexcept>

namespace thirdai::data {

TransformedTable::TransformedTable(thirdai::data::ColumnMap table,
                                   const TransformConfig& transform_config,
                                   thirdai::data::State& state)
    : table(transform_config.transform->apply(std::move(table), state)),
      inputs(transform_config.input_columns),
      labels(transform_config.label_columns) {}

void TransformedTable::removeIntermediateColumns() {
  std::unordered_map<std::string, thirdai::data::ColumnPtr> new_columns;
  for (const auto& column : inputs) {
    new_columns[column.indices()] = table.getColumn(column.indices());
    if (column.values()) {
      new_columns[*column.values()] = table.getColumn(*column.values());
    }
  }
  if (labels) {
    for (const auto& column : *labels) {
      new_columns[column.indices()] = table.getColumn(column.indices());
      if (column.values()) {
        new_columns[*column.values()] = table.getColumn(*column.values());
      }
    }
  }

  table = thirdai::data::ColumnMap(std::move(new_columns));
}

TransformedIterator::TransformedIterator(
    thirdai::data::ColumnMapIteratorPtr iter,
    const TransformConfig& transform_config, thirdai::data::StatePtr state)
    : iter(thirdai::data::TransformIterator::make(
          /* iter= */ std::move(iter),
          /* transform= */ transform_config.transform,
          /* state= */ std::move(state))),
      inputs(transform_config.input_columns),
      labels(transform_config.label_columns) {}

TransformedTensors::TransformedTensors(const TransformedTable& table) {
  inputs = thirdai::data::toTensors(table.table, table.inputs);
  if (table.labels) {
    labels = thirdai::data::toTensors(table.table, *table.labels);
  }
}

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
    if (column_info.values()) {
      values = columns.getArrayColumn<float>(*column_info.values());
    } else {
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
                                 ? 1.0 / indices_row.size()
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

bolt::LabeledDataset toLabeledDataset(const TransformedTable& table,
                                      size_t batch_size) {
  auto data = thirdai::data::toTensorBatches(
      /* columns= */ table.table,
      /* columns_to_convert= */ table.inputs,
      /* batch_size= */ batch_size);
  auto labels = thirdai::data::toTensorBatches(
      /* columns= */ table.table,
      /* columns_to_convert= */ table.labels.value(),
      /* batch_size= */ batch_size);

  return std::make_pair(std::move(data), std::move(labels));
}

}  // namespace thirdai::data
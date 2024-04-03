#include "TensorConversion.h"
#include <bolt/src/train/trainer/Dataset.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
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

ar::ConstArchivePtr outputColumnsToArchive(
    const OutputColumnsList& output_columns) {
  auto list = ar::List::make();
  for (const auto& output_column : output_columns) {
    auto map = ar::Map::make();
    map->set("indices", ar::str(output_column.indices()));
    if (output_column.values()) {
      map->set("values", ar::str(*output_column.values()));
    } else {
      switch (output_column.valueFillType()) {
        case ValueFillType::Ones:
          map->set("value_fill_type", ar::str("ones"));
          break;
        case ValueFillType::SumToOne:
          map->set("value_fill_type", ar::str("sum_to_one"));
          break;
        default:
          throw std::runtime_error(
              "Unsupported ValueFillType encountered in toArchive.");
      }
    }

    list->append(map);
  }

  return list;
}

OutputColumnsList outputColumnsFromArchive(const ar::Archive& archive) {
  OutputColumnsList output_columns;
  for (const auto& ar : archive.list()) {
    if (ar->contains("values")) {
      output_columns.emplace_back(ar->str("indices"), ar->str("values"));
    } else {
      std::string fill_type_name = ar->str("value_fill_type");
      ValueFillType value_fill_type;
      if (fill_type_name == "ones") {
        value_fill_type = ValueFillType::Ones;
      } else if (fill_type_name == "sum_to_one") {
        value_fill_type = ValueFillType::SumToOne;
      } else {
        throw std::runtime_error("Unsupported ValueFillType '" +
                                 fill_type_name +
                                 "' encountered in fromArchive.");
      }
      output_columns.emplace_back(ar->str("indices"), value_fill_type);
    }
  }
  return output_columns;
}

bolt::LabeledDataset toLabeledDataset(const ColumnMap& columns,
                                      const OutputColumnsList& input_columns,
                                      const OutputColumnsList& label_columns,
                                      size_t batch_size) {
  auto inputs = toTensorBatches(columns, input_columns, batch_size);
  auto labels = toTensorBatches(columns, label_columns, batch_size);
  return std::make_pair(std::move(inputs), std::move(labels));
}

}  // namespace thirdai::data
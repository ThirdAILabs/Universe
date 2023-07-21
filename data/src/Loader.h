#pragma once

#include <bolt/src/train/trainer/Dataset.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Transformation.h>
#include <optional>
#include <stdexcept>

namespace thirdai::data {

class Loader {
 public:
  Loader(ColumnMapIterator data_iterator, TransformationPtr transformation,
         IndexValueColumnList input_columns, IndexValueColumnList label_columns,
         size_t batch_size, size_t max_batches, size_t shuffle_buffer_size);

  std::optional<bolt::train::LabeledDataset> next();

 private:
  std::pair<ColumnMap, ColumnMap> splitIntoDataAndBuffer(
      ColumnMap&& loaded_rows, size_t dataset_size);

  ColumnMapIterator _data_iterator;
  TransformationPtr _transformation;

  std::vector<std::pair<std::string, std::string>> _input_columns;
  std::vector<std::pair<std::string, std::string>> _label_columns;

  size_t _batch_size;
  size_t _max_batches;
  size_t _shuffle_buffer_size;

  ColumnMap _shuffle_buffer;
};

}  // namespace thirdai::data
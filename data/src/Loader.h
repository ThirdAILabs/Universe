#pragma once

#include <bolt/src/train/trainer/Dataset.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Transformation.h>
#include <limits>
#include <optional>
#include <stdexcept>

namespace thirdai::data {

class Loader {
 public:
  static constexpr size_t NO_LIMIT = std::numeric_limits<size_t>::max();
  static constexpr size_t DEFAULT_SHUFFLE_BUFFER_SIZE = 64000;

  Loader(ColumnMapIterator data_iterator, TransformationPtr transformation,
         IndexValueColumnList input_columns, IndexValueColumnList label_columns,
         size_t batch_size, size_t max_batches = NO_LIMIT,
         size_t shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE);

  std::optional<bolt::train::LabeledDataset> next();

 private:
  std::pair<ColumnMap, ColumnMap> splitIntoDataAndBuffer(
      ColumnMap&& loaded_rows, size_t dataset_size) const;

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
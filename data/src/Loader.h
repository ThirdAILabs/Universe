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
         StatePtr state, IndexValueColumnList input_columns,
         IndexValueColumnList label_columns,
         size_t shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE,
         bool verbose = true);

  static auto make(ColumnMapIterator data_iterator,
                   TransformationPtr transformation, StatePtr state,
                   IndexValueColumnList input_columns,
                   IndexValueColumnList label_columns, size_t batch_size,
                   size_t max_batches = NO_LIMIT,
                   size_t shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE,
                   bool verbose = true) {
    return std::make_shared<Loader>(data_iterator, transformation, state,
                                    input_columns, label_columns, batch_size,
                                    max_batches, shuffle_buffer_size, verbose);
  }

  std::optional<bolt::train::LabeledDataset> next(
      size_t batch_size, size_t max_batches = NO_LIMIT);

  bolt::train::LabeledDataset all(size_t batch_size);

  void restart();

 private:
  std::pair<ColumnMap, ColumnMap> splitIntoDataAndBuffer(
      ColumnMap&& loaded_rows, size_t dataset_size) const;

  void logLoadStart() const;

  void logLoadEnd(size_t vectors, size_t batches, int64_t time) const;

  ColumnMapIterator _data_iterator;
  TransformationPtr _transformation;

  IndexValueColumnList _input_columns;
  IndexValueColumnList _label_columns;

  size_t _shuffle_buffer_size;
  bool _verbose;

  ColumnMap _shuffle_buffer;

  StatePtr _state;
};

using LoaderPtr = std::shared_ptr<Loader>;

}  // namespace thirdai::data
#include "Loader.h"
#include <limits>

namespace thirdai::data {

Loader::Loader(ColumnMapIterator data_iterator,
               TransformationPtr transformation,
               IndexValueColumnList input_columns,
               IndexValueColumnList label_columns, size_t batch_size,
               size_t max_batches, size_t shuffle_buffer_size)
    : _data_iterator(std::move(data_iterator)),
      _transformation(std::move(transformation)),
      _input_columns(std::move(input_columns)),
      _label_columns(std::move(label_columns)),
      _batch_size(batch_size),
      _max_batches(max_batches),
      _shuffle_buffer_size(shuffle_buffer_size),
      _shuffle_buffer(_data_iterator.emptyColumnMap()) {}

std::optional<bolt::train::LabeledDataset> Loader::next() {
  size_t num_rows_to_load;
  if (_max_batches == NO_LIMIT) {
    num_rows_to_load = NO_LIMIT;
  } else {
    num_rows_to_load = _shuffle_buffer_size + _batch_size * _max_batches;
  }

  ColumnMap loaded_rows = std::move(_shuffle_buffer);

  while (loaded_rows.numRows() < num_rows_to_load) {
    auto chunk = _data_iterator.next();
    if (!chunk) {
      break;
    }

    ColumnMap processed_chunk = _transformation->apply(std::move(*chunk));

    if (loaded_rows.numRows() > 0) {
      loaded_rows = loaded_rows.concat(processed_chunk);
    } else {
      loaded_rows = std::move(processed_chunk);
    }
  }

  if (loaded_rows.numRows() == 0) {
    return std::nullopt;
  }

  loaded_rows.shuffle();  // TODO(Nicholas) option to specify seed.

  auto [dataset, new_buffer] = splitIntoDataAndBuffer(
      std::move(loaded_rows), _batch_size * _max_batches);

  _shuffle_buffer = std::move(new_buffer);

  auto inputs = convertToTensors(dataset, _input_columns, _batch_size);
  auto labels = convertToTensors(dataset, _label_columns, _batch_size);

  return std::make_pair(std::move(inputs), std::move(labels));
}

std::pair<ColumnMap, ColumnMap> Loader::splitIntoDataAndBuffer(
    ColumnMap&& loaded_rows, size_t dataset_size) const {
  if (loaded_rows.numRows() <= dataset_size) {
    return std::make_pair(std::move(loaded_rows),
                          _data_iterator.emptyColumnMap());
  }
  return loaded_rows.split(dataset_size);
}

}  // namespace thirdai::data
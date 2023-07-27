#include "Loader.h"
#include <bolt/src/utils/Timer.h>
#include <data/src/TensorConversion.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::data {

Loader::Loader(ColumnMapIterator data_iterator,
               TransformationPtr transformation, StatePtr state,
               IndexValueColumnList input_columns,
               IndexValueColumnList label_columns, size_t batch_size,
               bool shuffle, bool verbose, size_t shuffle_buffer_size,
               uint32_t shuffle_seed)
    : _data_iterator(std::move(data_iterator)),
      _transformation(std::move(transformation)),
      _input_columns(std::move(input_columns)),
      _label_columns(std::move(label_columns)),
      _batch_size(batch_size),
      _verbose(verbose),
      _shuffle(shuffle),
      _shuffle_buffer_size(shuffle_buffer_size),
      _rng(shuffle_seed),
      _shuffle_buffer(ColumnMap({})),
      _state(std::move(state)) {
  if (!_state) {
    _state = std::make_shared<State>();
  }

  recordReturnedColumns(_input_columns);
  recordReturnedColumns(_label_columns);
}

std::optional<bolt::train::LabeledDataset> Loader::next(size_t max_batches) {
  logLoadStart();
  bolt::utils::Timer timer;

  // Prevents overflow since sometimes we pass in max int to indicate loading
  // all batches.
  size_t num_rows_to_load, num_rows_to_return;
  if (max_batches == NO_LIMIT || _batch_size == NO_LIMIT) {
    num_rows_to_load = NO_LIMIT;
    num_rows_to_return = NO_LIMIT;
  } else {
    num_rows_to_load = _shuffle_buffer_size + _batch_size * max_batches;
    num_rows_to_return = NO_LIMIT;
  }

  ColumnMap loaded_rows = std::move(_shuffle_buffer);

  while (loaded_rows.numRows() < num_rows_to_load) {
    auto chunk = _data_iterator.next();
    if (!chunk) {
      break;
    }

    ColumnMap processed_chunk = removeIntermediateColumns(
        _transformation->apply(std::move(*chunk), *_state));

    if (loaded_rows.numRows() > 0) {
      loaded_rows = loaded_rows.concat(processed_chunk);
    } else {
      // This is to avoid copying the chunk when loaded_rows is empty.
      loaded_rows = std::move(processed_chunk);
    }
  }

  if (loaded_rows.numRows() == 0) {
    timer.stop();
    logLoadEnd(/* vectors= */ 0, /* batches= */ 0, /* time= */ timer.seconds());
    return std::nullopt;
  }

  if (_shuffle) {
    loaded_rows.shuffle(_rng());
  }

  auto [dataset, new_buffer] =
      splitIntoDataAndBuffer(std::move(loaded_rows), num_rows_to_return);

  _shuffle_buffer = std::move(new_buffer);

  auto inputs = toTensorBatches(dataset, _input_columns, _batch_size);
  auto labels = toTensorBatches(dataset, _label_columns, _batch_size);

  timer.stop();
  logLoadEnd(dataset.numRows(), inputs.size(), timer.seconds());

  return std::make_pair(std::move(inputs), std::move(labels));
}

bolt::train::LabeledDataset Loader::all() {
  auto result = next(NO_LIMIT);
  if (!result) {
    throw std::invalid_argument("Could not load data from '" +
                                _data_iterator.resourceName() + "'.");
  }

  return std::move(result.value());
}

void Loader::restart() { _data_iterator.restart(); }

void Loader::recordReturnedColumns(
    const IndexValueColumnList& index_value_columns) {
  for (const auto& [indices, values] : index_value_columns) {
    _columns_returned.insert(indices);
    if (values) {
      _columns_returned.insert(*values);
    }
  }
}

ColumnMap Loader::removeIntermediateColumns(ColumnMap&& columns) const {
  std::unordered_map<std::string, ColumnPtr> returned_columns;
  for (const auto& [name, column] : columns) {
    if (_columns_returned.count(name)) {
      returned_columns[name] = column;
    }
  }

  return ColumnMap(returned_columns);
}

std::pair<ColumnMap, ColumnMap> Loader::splitIntoDataAndBuffer(
    ColumnMap&& loaded_rows, size_t dataset_size) {
  if (loaded_rows.numRows() <= dataset_size) {
    return std::make_pair(std::move(loaded_rows), ColumnMap({}));
  }
  return loaded_rows.split(dataset_size);
}

void Loader::logLoadStart() const {
#if THIRDAI_EXPOSE_ALL
  if (_verbose) {
    std::cout << "loading data | source '" << _data_iterator.resourceName()
              << "'" << std::endl;
  }
#endif
}

void Loader::logLoadEnd(size_t vectors, size_t batches, int64_t time) const {
#if THIRDAI_EXPOSE_ALL
  if (_verbose) {
    std::cout << "loading data | source '" << _data_iterator.resourceName()
              << "' | vectors " << vectors << " | batches " << batches
              << " | time " << time << "s | complete\n"
              << std::endl;
  }
#endif
}

}  // namespace thirdai::data
#include "Loader.h"
#include <bolt/src/utils/Timer.h>
#include <data/src/SmxTensorConversion.h>
#include <data/src/TensorConversion.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::data {

Loader::Loader(ColumnMapIteratorPtr data_iterator,
               TransformationPtr transformation, StatePtr state,
               OutputColumnsList model_input_columns,
               OutputColumnsList model_label_columns, size_t batch_size,
               bool shuffle, bool verbose, size_t shuffle_buffer_size,
               uint32_t shuffle_seed)
    : _data_iterator(std::move(data_iterator)),
      _transformation(std::move(transformation)),
      _model_input_columns(std::move(model_input_columns)),
      _model_label_columns(std::move(model_label_columns)),
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

  recordReturnedColumns(_model_input_columns);
  recordReturnedColumns(_model_label_columns);
}

std::optional<ColumnMap> Loader::nextColumnMap(size_t max_batches) {
  auto [num_rows_to_load, num_rows_to_return] = determineLoadSize(max_batches);

  ColumnMap loaded_rows = std::move(_shuffle_buffer);

  while (loaded_rows.numRows() < num_rows_to_load) {
    auto chunk = _data_iterator->next();
    if (!chunk) {
      break;
    }

    ColumnMap processed_chunk =
        _transformation->apply(std::move(*chunk), *_state);
    processed_chunk = removeIntermediateColumns(std::move(processed_chunk));

    if (loaded_rows.numRows() > 0) {
      loaded_rows = loaded_rows.concat(processed_chunk);
    } else {
      // This is to avoid copying the chunk when loaded_rows is empty.
      loaded_rows = std::move(processed_chunk);
    }
  }

  if (loaded_rows.numRows() == 0) {
    return std::nullopt;
  }

  if (_shuffle) {
    loaded_rows.shuffle(_rng());
  }

  auto [dataset, new_buffer] =
      splitIntoDataAndBuffer(std::move(loaded_rows), num_rows_to_return);

  _shuffle_buffer = std::move(new_buffer);

  return std::make_optional(std::move(dataset));
}

std::optional<bolt::LabeledDataset> Loader::next(size_t max_batches) {
  logLoadStart();
  bolt::utils::Timer timer;

  auto dataset = nextColumnMap(max_batches);

  if (!dataset) {
    timer.stop();
    logLoadEnd(/* vectors= */ 0, /* batches= */ 0, /* time= */ timer.seconds());
    return std::nullopt;
  }

  auto inputs = toTensorBatches(*dataset, _model_input_columns, _batch_size);
  auto labels = toTensorBatches(*dataset, _model_label_columns, _batch_size);

  timer.stop();
  logLoadEnd(dataset->numRows(), inputs.size(), timer.seconds());

  return std::make_pair(std::move(inputs), std::move(labels));
}

std::optional<std::pair<SmxDataset, SmxDataset>> Loader::nextSmx(
    size_t max_batches) {
  logLoadStart();
  bolt::utils::Timer timer;

  auto dataset = nextColumnMap(max_batches);

  if (!dataset) {
    timer.stop();
    logLoadEnd(/* vectors= */ 0, /* batches= */ 0, /* time= */ timer.seconds());
    return std::nullopt;
  }

  auto inputs = toSmxTensorBatches(*dataset, _model_input_columns, _batch_size);
  auto labels = toSmxTensorBatches(*dataset, _model_label_columns, _batch_size);

  timer.stop();
  logLoadEnd(dataset->numRows(), inputs.size(), timer.seconds());

  return std::make_pair(std::move(inputs), std::move(labels));
}

bolt::LabeledDataset Loader::all() {
  auto result = next(NO_LIMIT);
  if (!result) {
    throw std::invalid_argument("Could not load data from '" +
                                _data_iterator->resourceName() + "'.");
  }

  return std::move(result.value());
}

std::pair<SmxDataset, SmxDataset> Loader::allSmx() {
  auto result = nextSmx(NO_LIMIT);
  if (!result) {
    throw std::invalid_argument("Could not load data from '" +
                                _data_iterator->resourceName() + "'.");
  }

  return std::move(result.value());
}

void Loader::restart() { _data_iterator->restart(); }

void Loader::recordReturnedColumns(
    const OutputColumnsList& index_value_columns) {
  for (const auto& column : index_value_columns) {
    _columns_returned.insert(column.indices());
    if (column.values()) {
      _columns_returned.insert(*column.values());
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

std::pair<size_t, size_t> Loader::determineLoadSize(size_t max_batches) const {
  // This is to prevent overflow. We want the following to avoid overflow:
  // (batch_size * max_batches) + shuffle_buffer_size < INT_MAX
  // We can rewrite this to avoid computations that will result in overflow:
  // (INT_MAX - shuffle_buffer_size) / batch_size < max_batches

  if ((NO_LIMIT - _shuffle_buffer_size) / _batch_size < max_batches) {
    return {/*num_rows_to_load=*/NO_LIMIT, /*num_rows_to_return=*/NO_LIMIT};
  }
  return {/*num_rows_to_load=*/max_batches * _batch_size + _shuffle_buffer_size,
          /*num_rows_to_return=*/max_batches * _batch_size};
}

void Loader::logLoadStart() const {
#if THIRDAI_EXPOSE_ALL
  if (_verbose) {
    std::cout << "loading data | source '" << _data_iterator->resourceName()
              << "'" << std::endl;
  }
#else
  (void)_verbose;
#endif
}

void Loader::logLoadEnd(size_t vectors, size_t batches, double time) const {
#if THIRDAI_EXPOSE_ALL
  if (_verbose) {
    std::cout << "loading data | source '" << _data_iterator->resourceName()
              << "' | vectors " << vectors << " | batches " << batches
              << " | time " << time << "s | complete\n"
              << std::endl;
  }
#else
  (void)_verbose;
  (void)vectors;
  (void)batches;
  (void)time;
#endif
}

}  // namespace thirdai::data
#include "Loader.h"
#include <bolt/src/utils/Timer.h>
#include <data/src/TensorConversion.h>
#include <limits>
#include <stdexcept>

namespace thirdai::data {

Loader::Loader(ColumnMapIterator data_iterator,
               TransformationPtr transformation, StatePtr state,
               IndexValueColumnList input_columns,
               IndexValueColumnList label_columns, bool verbose,
               size_t shuffle_buffer_size, uint32_t shuffle_seed)
    : _data_iterator(std::move(data_iterator)),
      _transformation(std::move(transformation)),
      _input_columns(std::move(input_columns)),
      _label_columns(std::move(label_columns)),
      _shuffle_buffer_size(shuffle_buffer_size),
      _verbose(verbose),
      _shuffle_buffer(ColumnMap({})),
      _rng(shuffle_seed),
      _state(std::move(state)) {
  if (!_state) {
    _state = std::make_shared<State>();
  }
}

std::optional<bolt::train::LabeledDataset> Loader::next(size_t batch_size,
                                                        size_t max_batches) {
  logLoadStart();
  bolt::utils::Timer timer;

  // Prevents overflow since sometimes we pass in max int to indicate loading
  // all batches.
  size_t num_rows_to_load;
  if (max_batches == NO_LIMIT || batch_size == NO_LIMIT) {
    num_rows_to_load = NO_LIMIT;
  } else {
    num_rows_to_load = _shuffle_buffer_size + batch_size * max_batches;
  }

  ColumnMap loaded_rows = std::move(_shuffle_buffer);

  while (loaded_rows.numRows() < num_rows_to_load) {
    auto chunk = _data_iterator.next();
    if (!chunk) {
      break;
    }

    ColumnMap processed_chunk =
        _transformation->apply(std::move(*chunk), *_state);

    if (loaded_rows.numRows() > 0) {
      loaded_rows = loaded_rows.concat(processed_chunk);
    } else {
      loaded_rows = std::move(processed_chunk);
    }
  }

  if (loaded_rows.numRows() == 0) {
    timer.stop();
    logLoadEnd(0, 0, timer.seconds());
    return std::nullopt;
  }

  loaded_rows.shuffle(_rng());

  auto [dataset, new_buffer] =
      splitIntoDataAndBuffer(std::move(loaded_rows), batch_size * max_batches);

  _shuffle_buffer = std::move(new_buffer);

  auto inputs = toTensorBatches(dataset, _input_columns, batch_size);
  auto labels = toTensorBatches(dataset, _label_columns, batch_size);

  timer.stop();
  logLoadEnd(dataset.numRows(), inputs.size(), timer.seconds());

  return std::make_pair(std::move(inputs), std::move(labels));
}

bolt::train::LabeledDataset Loader::all(size_t batch_size) {
  auto result = next(batch_size, NO_LIMIT);
  if (!result) {
    throw std::invalid_argument("Could not load data from '" +
                                _data_iterator.resourceName() + "'.");
  }

  return std::move(result.value());
}

void Loader::restart() { _data_iterator.restart(); }

void Loader::addToShuffleBuffer(ColumnMap&& columns) {
  if (_shuffle_buffer.numRows() == 0) {
    _shuffle_buffer = std::move(columns);
  } else {
    _shuffle_buffer.concat(columns);
  }
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
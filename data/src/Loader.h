#pragma once

#include <bolt/src/train/trainer/Dataset.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <utils/Random.h>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace thirdai::data {

class Loader {
 public:
  static constexpr size_t NO_LIMIT = std::numeric_limits<size_t>::max();
  static constexpr size_t DEFAULT_SHUFFLE_BUFFER_SIZE = 64000;

  Loader(ColumnMapIteratorPtr data_iterator, TransformationPtr transformation,
         StatePtr state, OutputColumnsList input_columns,
         OutputColumnsList label_columns, size_t batch_size, bool shuffle,
         bool verbose = true,
         size_t shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE,
         uint32_t shuffle_seed = global_random::nextSeed());

  static auto make(ColumnMapIteratorPtr data_iterator,
                   TransformationPtr transformation, StatePtr state,
                   OutputColumnsList input_columns,
                   OutputColumnsList label_columns, size_t batch_size,
                   bool shuffle, bool verbose = true,
                   size_t shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE,
                   uint32_t shuffle_seed = global_random::nextSeed()) {
    return std::make_shared<Loader>(
        std::move(data_iterator), std::move(transformation), std::move(state),
        std::move(input_columns), std::move(label_columns), batch_size, shuffle,
        verbose, shuffle_buffer_size, shuffle_seed);
  }

  static auto make(ColumnMapIteratorPtr data_iterator,
                   OutputColumnsList input_columns,
                   OutputColumnsList label_columns, size_t batch_size,
                   bool shuffle, bool verbose = true,
                   size_t shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE,
                   uint32_t shuffle_seed = global_random::nextSeed()) {
    return std::make_shared<Loader>(
        std::move(data_iterator),
        /* transformation= */ TransformationList::make({}),
        /* state= */ nullptr, std::move(input_columns),
        std::move(label_columns), batch_size, shuffle, verbose,
        shuffle_buffer_size, shuffle_seed);
  }

  std::optional<bolt::LabeledDataset> next(size_t max_batches = NO_LIMIT);

  bolt::LabeledDataset all();

  void restart();

 private:
  void recordReturnedColumns(const OutputColumnsList& index_value_columns);

  ColumnMap removeIntermediateColumns(ColumnMap&& columns) const;

  static std::pair<ColumnMap, ColumnMap> splitIntoDataAndBuffer(
      ColumnMap&& loaded_rows, size_t dataset_size);

  std::pair<size_t, size_t> determineLoadSize(size_t max_batches) const;

  void logLoadStart() const;

  void logLoadEnd(size_t vectors, size_t batches, double time) const;

  ColumnMapIteratorPtr _data_iterator;
  TransformationPtr _transformation;

  OutputColumnsList _model_input_columns;
  OutputColumnsList _model_label_columns;
  std::unordered_set<std::string> _columns_returned;

  size_t _batch_size;
  bool _verbose;
  bool _shuffle;
  size_t _shuffle_buffer_size;
  std::mt19937 _rng;

  ColumnMap _shuffle_buffer;

  StatePtr _state;
};

using LoaderPtr = std::shared_ptr<Loader>;

}  // namespace thirdai::data
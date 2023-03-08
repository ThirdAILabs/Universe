#include "TidyBatcher.h"

namespace thirdai::dataset {
void TidyBatcher::add(std::vector<std::vector<BoltVector>>&& columns) {
  if (columns.empty()) {
    return;
  }

  if (columns.size() != _num_columns) {
    throw std::invalid_argument("Added vectors with wrong number of columns.");
  }

  size_t num_vectors = columns.front().size();
  size_t old_buffer_size = _buffer.size();
  _buffer.resize(_buffer.size() + num_vectors);

#pragma omp parallel for default(none) \
    shared(num_vectors, _buffer, old_buffer_size, columns)
  for (size_t vec_id = 0; vec_id < num_vectors; vec_id++) {
    _buffer[old_buffer_size + vec_id].resize(columns.size());
    for (size_t column_id = 0; column_id < columns.size(); column_id++) {
      _buffer[old_buffer_size + vec_id][column_id] =
          std::move(columns[column_id][vec_id]);
    }
  }
}

BatchColumns TidyBatcher::pop(size_t max_num_batches, size_t batch_size) {
  if (_buffer.empty()) {
    return BatchColumns(_num_columns);
  }

  if (_shuffle) {
    std::shuffle(_buffer.begin(), _buffer.end(), _gen);
  }

  size_t num_batches_in_buffer = (_buffer.size() + batch_size - 1) / batch_size;
  size_t num_batches = std::min(max_num_batches, num_batches_in_buffer);
  size_t last_batch_size = _buffer.size() % batch_size;
  size_t num_columns = _buffer.front().size();

  std::vector<std::vector<std::vector<BoltVector>>> batch_vectors(num_batches);

#pragma omp parallel for default(none) \
    shared(batch_vectors, num_batches, last_batch_size, batch_size)
  for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
    size_t this_batch_size =
        batch_id == num_batches_in_buffer - 1 ? last_batch_size : batch_size;
    batch_vectors[batch_id] = std::vector<std::vector<BoltVector>>(
        num_columns, std::vector<BoltVector>(this_batch_size));
  }

  BatchColumns batches(num_columns, std::vector<BoltBatch>(num_batches));
  for (size_t batch_id = 0; batch_id < num_batches; batch_id++) {
    size_t this_batch_size =
        batch_id == num_batches_in_buffer - 1 ? last_batch_size : batch_size;

    // std::vector<std::vector<BoltVector>> batch_vectors(
    //     num_columns, std::vector<BoltVector>(this_batch_size));

#pragma omp parallel for default(none) \
    shared(this_batch_size, num_columns, batch_vectors, _buffer)
    for (size_t vec_id = 0; vec_id < this_batch_size; vec_id++) {
      for (size_t column_id = 0; column_id < num_columns; column_id++) {
        batch_vectors[column_id][vec_id] =
            std::move(_buffer[vec_id][column_id]);
      }
    }
    _buffer.erase(_buffer.begin(), _buffer.begin() + this_batch_size);

    for (size_t column_id = 0; column_id < num_columns; column_id++) {
      batches[column_id][batch_id] =
          BoltBatch(std::move(batch_vectors[column_id]));
    }
  }

  return batches;
}
}  // namespace thirdai::dataset
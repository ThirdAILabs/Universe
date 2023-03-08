#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>
namespace thirdai::dataset {

using BatchColumns = std::vector<std::vector<BoltBatch>>;

static std::vector<size_t> permutation(size_t size, std::mt19937& gen) {
  std::vector<size_t> ordering(size);
  std::iota(ordering.begin(), ordering.end(), 0);
  std::shuffle(ordering.begin(), ordering.end(), gen);
  return ordering;
}

static void shuffleInPlace(BatchColumns& columns, std::mt19937& gen) {
  if (columns.empty()                     // No columns
      || columns.front().empty()          // Columns are empty
      || columns.front().front().empty()  // Batches are empty
  ) {
    return;
  }

  size_t num_batches = columns.front().size();
  size_t batch_size = columns.front().front().getBatchSize();
  size_t last_batch_size = columns.front().back().getBatchSize();
  size_t num_vectors = (num_batches - 1) * batch_size + last_batch_size;

  auto ordering = permutation(num_vectors, gen);
  std::vector<bool> swapped(num_vectors, false);

  for (size_t batch_id = 0; batch_id < num_batches; batch_id++) {
    size_t this_batch_size = columns.front().at(batch_id).getBatchSize();
#pragma omp parallel for default(none) \
    shared(this_batch_size, batch_id, batch_size, ordering, columns)
    for (size_t vec_id = 0; vec_id < this_batch_size; vec_id++) {
      size_t id = batch_id * batch_size + vec_id;
      size_t swap_id = ordering[id];
      size_t swap_batch_id = swap_id / batch_size;

      if (swap_batch_id != batch_id && !swapped[id]) {
        size_t swap_vec_id = swap_id % batch_size;
        for (auto& column : columns) {
          std::swap(column[batch_id][vec_id],
                    column[swap_batch_id][swap_vec_id]);
          swapped[swap_id] = true;
        }
      }
    }
  }
}

}  // namespace thirdai::dataset
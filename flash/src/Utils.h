#pragma once

#include <numeric>
#include <queue>
#include <vector>

namespace thirdai::search {

template <class T>
void removeDuplicates(std::vector<T>& v) {
  std::sort(v.begin(), v.end());
  v.erase(unique(v.begin(), v.end()), v.end());
}

// Finds the top_k largest indices in a vector using a priority queue.
template <class T>
std::vector<uint32_t> argmax(const std::vector<T>& input, uint32_t top_k) {
  static_assert(std::is_signed<T>::value,
                "The input to the argmax needs to be signed so we can negate "
                "the values for the priority queue.");
  std::priority_queue<std::pair<T, uint32_t>> min_heap;
  for (uint32_t i = 0; i < input.size(); i++) {
    if (min_heap.size() < top_k) {
      min_heap.emplace(-input[i], i);
    } else if (-input[i] < min_heap.top().first) {
      min_heap.pop();
      min_heap.emplace(-input[i], i);
    }
  }

  std::vector<uint32_t> result;
  while (!min_heap.empty()) {
    result.push_back(min_heap.top().second);
    min_heap.pop();
  }

  std::reverse(result.begin(), result.end());

  return result;
}

// Performs an argsort on the input vector. The sort is descending, e.g. the
// index of the largest element in to_argsort is the first element in the
// result. to_argsort should be a vector of size less than UINT32_MAX.
template <class T>
std::vector<uint32_t> argsort_descending(const std::vector<T> to_argsort) {
  std::vector<uint32_t> indices(to_argsort.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(),
                   [&to_argsort](size_t i1, size_t i2) {
                     return to_argsort[i1] > to_argsort[i2];
                   });
  return indices;
}

}  // namespace thirdai::search
#include "Flash.h"
#include <hashtable/src/SampledHashTable.h>
#include <hashtable/src/VectorHashTable.h>
#include <algorithm>
#include <queue>
#include <vector>

namespace thirdai::search {

template class Flash<uint32_t>;
template class Flash<uint64_t>;

template <typename LABEL_T>
Flash<LABEL_T>::Flash(const hashing::HashFunction& function)
    : _function(function),
      _num_tables(_function.numTables()),
      _range(_function.range()),
      _hashtable(new hashtable::VectorHashTable<LABEL_T, false>(_num_tables,
                                                                _range)) {}

template <typename LABEL_T>
Flash<LABEL_T>::Flash(const hashing::HashFunction& function,
                      uint32_t reservoir_size)
    : _function(function),
      _num_tables(_function.numTables()),
      _range(_function.range()),
      _hashtable(new hashtable::VectorHashTable<LABEL_T, true>(
          _num_tables, reservoir_size, _range)) {}

template void Flash<uint32_t>::addDataset<dataset::SparseBatch>(
    dataset::InMemoryDataset<dataset::SparseBatch>&);
template void Flash<uint64_t>::addDataset<dataset::SparseBatch>(
    dataset::InMemoryDataset<dataset::SparseBatch>&);

template void Flash<uint32_t>::addDataset<dataset::DenseBatch>(
    dataset::InMemoryDataset<dataset::DenseBatch>&);
template void Flash<uint64_t>::addDataset<dataset::DenseBatch>(
    dataset::InMemoryDataset<dataset::DenseBatch>&);

template <typename LABEL_T>
template <typename BATCH_T>
void Flash<LABEL_T>::addDataset(dataset::InMemoryDataset<BATCH_T>& dataset) {
  for (uint64_t batch_id = 0; batch_id < dataset.numBatches(); batch_id++) {
    addBatch(dataset[batch_id]);
  }
}

template void Flash<uint32_t>::addDataset<dataset::SparseBatch>(
    dataset::StreamedDataset<dataset::SparseBatch>&);
template void Flash<uint64_t>::addDataset<dataset::SparseBatch>(
    dataset::StreamedDataset<dataset::SparseBatch>&);

template void Flash<uint32_t>::addDataset<dataset::DenseBatch>(
    dataset::StreamedDataset<dataset::DenseBatch>&);
template void Flash<uint64_t>::addDataset<dataset::DenseBatch>(
    dataset::StreamedDataset<dataset::DenseBatch>&);

template <typename LABEL_T>
template <typename BATCH_T>
void Flash<LABEL_T>::addDataset(dataset::StreamedDataset<BATCH_T>& dataset) {
  while (auto batch = dataset.nextBatch()) {
    addBatch(*batch);
  }
}

template void Flash<uint32_t>::addBatch<dataset::SparseBatch>(
    const dataset::SparseBatch&);
template void Flash<uint64_t>::addBatch<dataset::SparseBatch>(
    const dataset::SparseBatch&);

template void Flash<uint32_t>::addBatch<dataset::DenseBatch>(
    const dataset::DenseBatch&);
template void Flash<uint64_t>::addBatch<dataset::DenseBatch>(
    const dataset::DenseBatch&);

template <typename LABEL_T>
template <typename BATCH_T>
void Flash<LABEL_T>::addBatch(const BATCH_T& batch) {
  uint32_t* hashes = hash(batch);
  try {
    verifyBatchSequentialIds(batch);
  } catch (std::invalid_argument& e) {
    delete[] hashes;
    throw e;
  }
  _hashtable->insertSequential(batch.getBatchSize(), batch.id(0), hashes);

  delete[] hashes;
}

template <typename LABEL_T>
template <typename BATCH_T>
uint32_t* Flash<LABEL_T>::hash(const BATCH_T& batch) const {
  uint32_t* hashes = new uint32_t[batch.getBatchSize() * _num_tables];
  _function.hashBatchParallel(batch, hashes);
  return hashes;
}

template <typename LABEL_T>
template <typename BATCH_T>
void Flash<LABEL_T>::verifyBatchSequentialIds(const BATCH_T& batch) const {
  uint64_t largest_batch_id = batch.id(0) + batch.getBatchSize();
  verify_and_convert_id(largest_batch_id);
}

template <typename LABEL_T>
LABEL_T Flash<LABEL_T>::verify_and_convert_id(uint64_t id) const {
  // Casting to a smaller integer is well specified behavior because we are
  // dealing with only unsigned integers. If the largest_batch_id is out
  // of range of LABEL_T, its first bits will get truncated and the equality
  // check will fail (we cast back to uin64_t to ensure that the
  // largest_batch_id itself is not casst down to LABEL_T).
  LABEL_T cast_id = static_cast<LABEL_T>(id);
  bool out_of_range = static_cast<uint64_t>(cast_id) != id;
  if (out_of_range) {
    throw std::invalid_argument("Trying to insert vector with id " +
                                std::to_string(id) +
                                ", which is too large an id for this Flash.");
  }
  return cast_id;
}

template std::vector<std::vector<uint32_t>>
Flash<uint32_t>::queryBatch<dataset::SparseBatch>(const dataset::SparseBatch&,
                                                  uint32_t, bool) const;
template std::vector<std::vector<uint64_t>>
Flash<uint64_t>::queryBatch<dataset::SparseBatch>(const dataset::SparseBatch&,
                                                  uint32_t, bool) const;

template std::vector<std::vector<uint32_t>>
Flash<uint32_t>::queryBatch<dataset::DenseBatch>(const dataset::DenseBatch&,
                                                 uint32_t, bool) const;
template std::vector<std::vector<uint64_t>>
Flash<uint64_t>::queryBatch<dataset::DenseBatch>(const dataset::DenseBatch&,
                                                 uint32_t, bool) const;

template <typename LABEL_T>
template <typename BATCH_T>
std::vector<std::vector<LABEL_T>> Flash<LABEL_T>::queryBatch(
    const BATCH_T& batch, uint32_t top_k, bool pad_zeros) const {
  std::vector<std::vector<LABEL_T>> results(batch.getBatchSize());
  uint32_t* hashes = hash(batch);

#pragma omp parallel for default(none) \
    shared(batch, top_k, results, hashes, pad_zeros)
  for (uint64_t vec_id = 0; vec_id < batch.getBatchSize(); vec_id++) {
    std::vector<LABEL_T> query_result;
    _hashtable->queryByVector(hashes + vec_id * _num_tables, query_result);
    results.at(vec_id) = getTopKUsingPriorityQueue(query_result, top_k);
    if (pad_zeros) {
      while (results.at(vec_id).size() < top_k) {
        results.at(vec_id).push_back(0);
      }
    }
  }

  delete[] hashes;

  return results;
}

template <typename LABEL_T>
Flash<LABEL_T>::~Flash<LABEL_T>() {
  delete _hashtable;
}

template <typename LABEL_T>
std::vector<LABEL_T> Flash<LABEL_T>::getTopKUsingPriorityQueue(
    std::vector<LABEL_T>& query_result, uint32_t top_k) const {
  // We sort so counting is easy
  std::sort(query_result.begin(), query_result.end());

  // To make this a max queue, we insert all element counts multiplied by -1
  std::priority_queue<std::pair<int32_t, uint64_t>> top_k_queue;
  if (!query_result.empty()) {
    uint64_t current_element = query_result.at(0);
    uint32_t current_element_count = 0;
    for (auto element : query_result) {
      if (element == current_element) {
        current_element_count++;
      } else {
        top_k_queue.emplace(-current_element_count, current_element);
        if (top_k_queue.size() > top_k) {
          top_k_queue.pop();
        }
        current_element = element;
        current_element_count = 1;
      }
    }
    top_k_queue.emplace(-current_element_count, current_element);
    if (top_k_queue.size() > top_k) {
      top_k_queue.pop();
    }
  }

  // Create and save results vector
  std::vector<LABEL_T> result;
  while (!top_k_queue.empty()) {
    result.push_back(top_k_queue.top().second);
    top_k_queue.pop();
  }
  std::reverse(result.begin(), result.end());

  return result;
}

}  // namespace thirdai::search
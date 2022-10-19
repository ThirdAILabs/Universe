#include <wrappers/src/LicenseWrapper.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashtable/src/SampledHashTable.h>
#include <hashtable/src/VectorHashTable.h>
#include <dataset/src/InMemoryDataset.h>
#include <indexer/src/Flash.h>
#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

namespace thirdai::bolt {

template <typename LABEL_T>
Flash<LABEL_T>::Flash(std::shared_ptr<hashing::HashFunction> function)
    : _hash_function(std::move(function)),
      _num_tables(_hash_function->numTables()),
      _range(_hash_function->range()),
      _batch_elements_counter(0),
      _hashtable(
          new hashtable::VectorHashTable<LABEL_T, false>(_num_tables, _range)) {
  thirdai::licensing::LicenseWrapper::checkLicense();
}

template <typename LABEL_T>
Flash<LABEL_T>::Flash(std::shared_ptr<hashing::HashFunction> function, uint32_t reservoir_size)
    : _hash_function(std::move(function)),
      _num_tables(_hash_function->numTables()),
      _range(_hash_function->range()),
      _hashtable(new hashtable::VectorHashTable<LABEL_T, true>(
          _num_tables, reservoir_size, _range)) {}

template <typename LABEL_T>
void Flash<LABEL_T>::addDataset(
    const dataset::InMemoryDataset<BoltBatch>& dataset) {
  for (uint64_t batch_id = 0; batch_id < dataset.numBatches(); batch_id++) {
    addBatch(dataset[batch_id]);
  }
}

template <typename LABEL_T>
void Flash<LABEL_T>::addDataset(dataset::StreamingDataset<BoltBatch>& dataset) {
  while (auto batch_tuple = dataset.nextBatchTuple()) {
    const auto& batch = batch_tuple.value();
    // ignore the labels
    addBatch(std::get<0>(batch));
  }
}

template <typename LABEL_T>
void Flash<LABEL_T>::addBatch(const BoltBatch& batch) {
  std::vector<uint32_t> hashes = hashBatch(batch);
  try {
    verifyBatchSequentialIds(batch);
  } catch (std::invalid_argument& error) {
    throw error;
  }

  _hashtable->insertSequential(batch.getBatchSize(), _batch_elements_counter,
                               hashes.data());

  incrementBatchElementsCounter(batch.getBatchSize());
}

template <typename LABEL_T>
std::vector<uint32_t> Flash<LABEL_T>::hashBatch(const BoltBatch& batch) const {
  auto hashes = _hash_function->hashBatchParallel(batch);
  return hashes;
}

template <typename LABEL_T>
void Flash<LABEL_T>::verifyBatchSequentialIds(const BoltBatch& batch) const {
  uint64_t largest_batch_id = _batch_elements_counter + batch.getBatchSize();
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

template <typename LABEL_T>
std::vector<std::vector<LABEL_T>> Flash<LABEL_T>::queryBatch(
    const BoltBatch& batch, uint32_t top_k, bool pad_zeros) const {
  std::vector<std::vector<LABEL_T>> results(batch.getBatchSize());
  auto hashes = hashBatch(batch);

#pragma omp parallel for default(none) \
    shared(batch, top_k, results, hashes, pad_zeros)
  for (uint64_t vec_id = 0; vec_id < batch.getBatchSize(); vec_id++) {
    std::vector<LABEL_T> query_result;
    _hashtable->queryByVector(hashes.data() + vec_id * _num_tables,
                              query_result);
    results.at(vec_id) = getTopKUsingPriorityQueue(query_result, top_k);
    if (pad_zeros) {
      while (results.at(vec_id).size() < top_k) {
        results.at(vec_id).push_back(0);
      }
    }
  }

  return results;
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

template class Flash<uint32_t>;
template class Flash<uint64_t>;

}  // namespace thirdai::bolt

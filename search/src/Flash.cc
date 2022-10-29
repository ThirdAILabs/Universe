#include <wrappers/src/LicenseWrapper.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashtable/src/SampledHashTable.h>
#include <hashtable/src/VectorHashTable.h>
#include <_types/_uint32_t.h>
#include <dataset/src/InMemoryDataset.h>
#include <search/src/Flash.h>
#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

namespace thirdai::search {

template <typename LABEL_T>
Flash<LABEL_T>::Flash(std::shared_ptr<hashing::HashFunction> function)
    : _hash_function(std::move(function)),
      _num_tables(_hash_function->numTables()),
      _range(_hash_function->range()),
      _batch_elements_counter(0),
      _hashtable(std::make_shared<hashtable::VectorHashTable<LABEL_T, false>>(
          _num_tables, _range)) {
  thirdai::licensing::LicenseWrapper::checkLicense();
}

template <typename LABEL_T>
Flash<LABEL_T>::Flash(std::shared_ptr<hashing::HashFunction> function,
                      uint32_t reservoir_size)
    : _hash_function(std::move(function)),
      _num_tables(_hash_function->numTables()),
      _range(_hash_function->range()),
      _hashtable(std::make_shared<hashtable::VectorHashTable<LABEL_T, true>>(
          _num_tables, reservoir_size, _range)) {
  thirdai::licensing::LicenseWrapper::checkLicense();
}

template <typename LABEL_T>
void Flash<LABEL_T>::addDataset(
    const dataset::InMemoryDataset<BoltBatch>& dataset,
    const std::vector<std::vector<LABEL_T>>& labels) {
  for (uint64_t batch_index = 0; batch_index < dataset.numBatches();
       batch_index++) {
    const auto& batch = dataset[batch_index];

    addBatch(batch, labels[batch_index]);
  }
}

template <typename LABEL_T>
void Flash<LABEL_T>::addDataset(
    dataset::StreamingDataset<BoltBatch>& dataset,
    const std::vector<std::vector<LABEL_T>>& labels) {
  uint32_t batch_index = 0;
  while (auto batch_tuple = dataset.nextBatchTuple()) {
    const auto& batch = std::get<0>(batch_tuple.value());

    addBatch(batch, labels[batch_index]);
    batch_index++;
  }
}

template <typename LABEL_T>
void Flash<LABEL_T>::addBatch(const BoltBatch& batch,
                              const std::vector<LABEL_T>& labels) {
  std::vector<uint32_t> hashes = hashBatch(batch);

  assert(hashes.size() == batch.getBatchSize() * _num_tables);

  verifyIDFitsLabelTypeRange(batch.getBatchSize());
  _hashtable->insert(batch.getBatchSize(), labels.data(), hashes.data());

  incrementBatchElementsCounter(batch.getBatchSize());
}

template <typename LABEL_T>
std::vector<uint32_t> Flash<LABEL_T>::hashBatch(const BoltBatch& batch) const {
  auto hashes = _hash_function->hashBatchParallel(batch);
  return hashes;
}

template <typename LABEL_T>
void Flash<LABEL_T>::verifyIDFitsLabelTypeRange(uint64_t id) const {
  uint64_t max_possible_value = std::numeric_limits<LABEL_T>::max();

  if (id + _batch_elements_counter > max_possible_value) {
    throw std::invalid_argument(
        "Trying to insert vector with id " + std::to_string(id) +
        ", which is too large an id for this Flash Index.");
  }
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

  std::priority_queue<std::pair<int32_t, LABEL_T>,
                      std::vector<std::pair<int32_t, LABEL_T>>,
                      std::greater<std::pair<int32_t, LABEL_T>>>
      queue;

  if (!query_result.empty()) {
    uint64_t current_element = query_result.at(0);
    uint32_t current_element_count = 0;
    for (auto element : query_result) {
      if (element == current_element) {
        current_element_count++;
      } else {
        queue.emplace(current_element_count, current_element);
        if (queue.size() > top_k) {
          queue.pop();
        }
        current_element = element;
        current_element_count = 1;
      }
    }
    queue.emplace(current_element_count, current_element);
    if (queue.size() > top_k) {
      queue.pop();
    }
  }

  // Create and save results vector
  std::vector<LABEL_T> result;
  while (!queue.empty()) {
    result.push_back(queue.top().second);
    queue.pop();
  }
  std::reverse(result.begin(), result.end());

  return result;
}

template class Flash<uint32_t>;
template class Flash<uint64_t>;

}  // namespace thirdai::search

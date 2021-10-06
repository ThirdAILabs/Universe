#include "Flash.h"
#include "../../utils/hashtable/SampledHashTable.h"
#include "../../utils/hashtable/VectorHashTable.h"
#include <algorithm>
#include <queue>
#include <vector>

namespace thirdai::search {

template class Flash<uint8_t>;
template class Flash<uint16_t>;
template class Flash<uint32_t>;
template class Flash<uint64_t>;

template <typename Label_t>
Flash<Label_t>::Flash(const utils::HashFunction& function)
    : _function(function),
      _num_tables(_function.numTables()),
      _range(_function.range()),
      _hashtable(new utils::VectorHashTable<Label_t>(_num_tables, _range)) {}
// TODO(josh/nicholas): Figure out why the SampledHashTable doesn't work well
// _hashtable(new utils::SampledHashTable<Label_t>(_num_tables, 100, _range)) {}

template <typename Label_t>
void Flash<Label_t>::addDataset(utils::Dataset& dataset) {
  dataset.loadNextBatchSet();
  while (dataset.numBatches() > 0) {
    for (uint64_t batch_id = 0; batch_id < dataset.numBatches(); batch_id++) {
      addBatch(dataset[batch_id]);
    }
    dataset.loadNextBatchSet();
  }
}

template <typename Label_t>
void Flash<Label_t>::addBatch(const utils::Batch& batch) {
  uint32_t* hashes = hash(batch);
  switch (batch._id_type) {
    case thirdai::utils::ID_TYPE::SEQUENTIAL:
      verifyBatchSequentialIds(batch);
      _hashtable->insertSequential(batch._batch_size, batch._starting_id,
                                   hashes);
      break;
    case thirdai::utils::ID_TYPE::INDIVIDUAL:
      std::vector<Label_t> converted_ids = verifyBatchIndependentIds(batch);
      _hashtable->insert(batch._batch_size, converted_ids.data(), hashes);
      break;
  }
  delete hashes;
}

template <typename Label_t>
uint32_t* Flash<Label_t>::hash(const utils::Batch& batch) const {
  uint32_t* hashes = new uint32_t[batch._batch_size * _num_tables];
  _function.hashBatchParallel(batch, hashes);
  return hashes;
}

template <typename Label_t>
void Flash<Label_t>::verifyBatchSequentialIds(const utils::Batch& batch) const {
  uint64_t largest_batch_id = batch._starting_id + batch._batch_size;
  verify_and_convert_id(largest_batch_id);
}

template <typename Label_t>
std::vector<Label_t> Flash<Label_t>::verifyBatchIndependentIds(
    const utils::Batch& batch) const {
  std::vector<Label_t> converted_ids;
  for (uint32_t vec_id = 0; vec_id < batch._batch_size; vec_id++) {
    converted_ids.push_back(
        verify_and_convert_id(batch._individual_ids[vec_id]));
  }
  return converted_ids;
}

template <typename Label_t>
Label_t Flash<Label_t>::verify_and_convert_id(uint64_t id) const {
  // Casting to a smaller integer is well specified behavior because we are
  // dealing with only unsigned integers. If the largest_batch_id is out
  // of range of Label_t, its first bits will get truncated and the equality
  // check will fail (we cast back to uin64_t to ensure that the
  // largest_batch_id itself is not casst down to Label_t).
  Label_t cast_id = static_cast<Label_t>(id);
  bool out_of_range = static_cast<uint64_t>(cast_id) != id;
  if (out_of_range) {
    throw std::invalid_argument("Trying to insert vector with id " +
                                std::to_string(id) +
                                ", which is too large an id for this Flash.");
  }
  return cast_id;
}

template <typename Label_t>
std::vector<std::vector<Label_t>> Flash<Label_t>::queryBatch(
    const utils::Batch& batch, uint32_t top_k, bool pad_zeros) const {
  std::vector<std::vector<Label_t>> results(batch._batch_size);
  uint32_t* hashes = hash(batch);

#pragma omp parallel for default(none) \
    shared(batch, top_k, results, hashes, pad_zeros)
  for (uint64_t vec_id = 0; vec_id < batch._batch_size; vec_id++) {
    std::vector<Label_t> query_result;
    _hashtable->queryByVector(hashes + vec_id * _num_tables, query_result);
    results.at(vec_id) = getTopKUsingPriorityQueue(query_result, top_k);
    if (pad_zeros) {
      while (results.at(vec_id).size() < top_k) {
        results.at(vec_id).push_back(0);
      }
    }
  }

  delete hashes;

  return results;
}

template <typename Label_t>
std::vector<Label_t> Flash<Label_t>::getTopKUsingPriorityQueue(
    std::vector<Label_t>& query_result, uint32_t top_k) const {
  // We sort so counting is easy
  std::sort(query_result.begin(), query_result.end());

  // To make this a max queue, we insert all element counts multiplied by -1
  std::priority_queue<std::pair<uint32_t, uint64_t>> top_k_queue;
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
  std::vector<Label_t> result;
  while (!top_k_queue.empty()) {
    result.push_back(top_k_queue.top().second);
    top_k_queue.pop();
  }
  std::reverse(result.begin(), result.end());

  return result;
}

}  // namespace thirdai::search
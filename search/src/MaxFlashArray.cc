#include "MaxFlashArray.h"
#include <hashing/src/HashFunction.h>
#include <hashtable/src/HashTable.h>
#include <dataset/src/Dataset.h>
#include <cmath>
#include <utility>

namespace thirdai::search {

template class MaxFlashArray<uint8_t>;
template class MaxFlashArray<uint16_t>;
template class MaxFlashArray<uint32_t>;

template <typename LABEL_T>
MaxFlashArray<LABEL_T>::MaxFlashArray(hashing::HashFunction* function,
                                      uint32_t hashes_per_table,
                                      uint64_t max_doc_size)
    : _max_allowable_doc_size(std::min<uint64_t>(
          max_doc_size, std::numeric_limits<LABEL_T>::max())),
      _hash_function(function),
      _maxflash_array(),
      _collision_count_to_sim(function->range()) {
  for (uint32_t i = 0; i < _hash_function->numTables(); i++) {
    _collision_count_to_sim[i] =
        std::exp(std::log(static_cast<float>(i) / function->numTables()) /
                 hashes_per_table);
  }
}

template uint64_t MaxFlashArray<uint8_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&);
template uint64_t MaxFlashArray<uint16_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&);
template uint64_t MaxFlashArray<uint32_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&);

template uint64_t MaxFlashArray<uint8_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&);
template uint64_t MaxFlashArray<uint16_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&);
template uint64_t MaxFlashArray<uint32_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&);

template <typename LABEL_T>
template <typename BATCH_T>
uint64_t MaxFlashArray<LABEL_T>::addDocument(const BATCH_T& batch) {
  LABEL_T num_elements =
      std::min<uint64_t>(batch.getBatchSize(), _max_allowable_doc_size);
  const std::vector<uint32_t> hashes = hash(batch);
  _maxflash_array.push_back(std::make_unique<MaxFlash<LABEL_T>>(
      _hash_function->numTables(), _hash_function->range(), num_elements,
      hashes));
  return _maxflash_array.size() - 1;
}

template std::vector<float> MaxFlashArray<uint8_t>::getDocumentScores(
    const dataset::DenseBatch& query,
    const std::vector<uint32_t>& documents_to_query) const;
template std::vector<float> MaxFlashArray<uint16_t>::getDocumentScores(
    const dataset::DenseBatch& query,
    const std::vector<uint32_t>& documents_to_query) const;
template std::vector<float> MaxFlashArray<uint32_t>::getDocumentScores(
    const dataset::DenseBatch& query,
    const std::vector<uint32_t>& documents_to_query) const;

template <typename LABEL_T>
std::vector<float> MaxFlashArray<LABEL_T>::getDocumentScores(
    const dataset::DenseBatch& query,
    const std::vector<uint32_t>& documents_to_query) const {
  const std::vector<uint32_t> hashes = hash(query);

  std::vector<float> result(documents_to_query.size());

#pragma omp parallel default(none) \
    shared(result, documents_to_query, hashes, query)
  {
    std::vector<uint32_t> buffer(_max_allowable_doc_size);

#pragma omp for
    for (uint64_t i = 0; i < result.size(); i++) {
      uint64_t flash_index = documents_to_query.at(i);
      result[i] = _maxflash_array.at(flash_index)
                      ->getScore(hashes, query.getBatchSize(), buffer,
                                 _collision_count_to_sim);
    }
  }

  return result;
}

template <typename LABEL_T>
template <typename BATCH_T>
std::vector<uint32_t> MaxFlashArray<LABEL_T>::hash(const BATCH_T& batch) const {
  return _hash_function->hashBatchParallel(batch);
}

}  // namespace thirdai::search
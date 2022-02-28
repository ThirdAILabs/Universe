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
template class MaxFlashArray<uint64_t>;

template <typename LABEL_T>
MaxFlashArray<LABEL_T>::MaxFlashArray(hashing::HashFunction* function,
                                      uint32_t hashes_per_table,
                                      uint64_t max_doc_size)
    : _max_allowable_doc_size(std::min<uint64_t>(
          max_doc_size, std::numeric_limits<LABEL_T>::max())),
      _function(function),
      _maxflash_array(),
      _lookups(function->range())
{
  for (uint32_t i = 0; i < _function->numTables(); i++) {
    _lookups[i] =
        std::exp(std::log(static_cast<float>(i) / function->numTables()) /
                 hashes_per_table);
  }
}

template void MaxFlashArray<uint8_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&);
template void MaxFlashArray<uint16_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&);
template void MaxFlashArray<uint32_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&);
template void MaxFlashArray<uint64_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&);

template void MaxFlashArray<uint8_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&);
template void MaxFlashArray<uint16_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&);
template void MaxFlashArray<uint32_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&);
template void MaxFlashArray<uint64_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&);

template <typename LABEL_T>
template <typename BATCH_T>
void MaxFlashArray<LABEL_T>::addDocument(const BATCH_T& batch) {
  LABEL_T num_elements =
      std::min<uint64_t>(batch.getBatchSize(), _max_allowable_doc_size);
  uint32_t* hashes = hash(batch);
  _maxflash_array.push_back(std::make_unique<MaxFlash<LABEL_T>>(
      _function->numTables(), _function->range(), num_elements, hashes));
  delete[] hashes;
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
template std::vector<float> MaxFlashArray<uint64_t>::getDocumentScores(
    const dataset::DenseBatch& query,
    const std::vector<uint32_t>& documents_to_query) const;

template <typename LABEL_T>
std::vector<float> MaxFlashArray<LABEL_T>::getDocumentScores(
    const dataset::DenseBatch& query,
    const std::vector<uint32_t>& documents_to_query) const {
  uint32_t* hashes = hash(query);

  std::vector<float> result(documents_to_query.size());

#pragma omp parallel default(none) \
    shared(result, documents_to_query, hashes, query)
  {
    std::vector<uint32_t> buffer(_max_allowable_doc_size);

#pragma omp for
    for (uint64_t i = 0; i < result.size(); i++) {
      uint64_t flash_index = documents_to_query.at(i);
      result[i] =
          _maxflash_array.at(flash_index)
              ->getScore(hashes, query.getBatchSize(), buffer, _lookups);
    }
  }

  delete[] hashes;
  return result;
}

/**
 * Returns a pointer to the hashes of the input batch. These hashes will need
 * to be deleted by the calling function.
 */

template <typename LABEL_T>
template <typename BATCH_T>
uint32_t* MaxFlashArray<LABEL_T>::hash(const BATCH_T& batch) const {
  uint32_t* hashes =
      new uint32_t[batch.getBatchSize() * _function->numTables()];
  // TODO(josh): Parallilize across documents
  // _function->hashBatchSerial(batch, hashes);
  _function->hashBatchParallel(batch, hashes);
  return hashes;
}

}  // namespace thirdai::search
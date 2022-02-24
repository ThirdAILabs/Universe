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
                                      uint64_t num_flashes,
                                      uint32_t hashes_per_table)
    : _function(function),
      _maxflash_array(),
      _lookups(function->range()),
      _largest_doc(0) {
  for (uint64_t i = 0; i < num_flashes; i++) {
    _maxflash_array.push_back(
        new MaxFlash<LABEL_T>(_function->numTables(), _function->range()));
  }
  for (uint32_t i = 0; i < _function->numTables(); i++) {
    _lookups[i] =
        std::exp(std::log(static_cast<float>(i) / function->numTables()) /
                 hashes_per_table);
  }
}

template void MaxFlashArray<uint8_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&, uint64_t);
template void MaxFlashArray<uint16_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&, uint64_t);
template void MaxFlashArray<uint32_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&, uint64_t);
template void MaxFlashArray<uint64_t>::addDocument<dataset::SparseBatch>(
    const dataset::SparseBatch&, uint64_t);

template void MaxFlashArray<uint8_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&, uint64_t);
template void MaxFlashArray<uint16_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&, uint64_t);
template void MaxFlashArray<uint32_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&, uint64_t);
template void MaxFlashArray<uint64_t>::addDocument<dataset::DenseBatch>(
    const dataset::DenseBatch&, uint64_t);

// TODO(josh): Make this safer to parallilize
template <typename LABEL_T>
template <typename BATCH_T>
void MaxFlashArray<LABEL_T>::addDocument(const BATCH_T& batch,
                                         uint64_t document_id) {
  _largest_doc = std::max(_largest_doc, batch.getBatchSize());
  uint32_t* hashes = hash(batch);
  _maxflash_array[document_id]->populate(
      hashes, std::min<uint64_t>(batch.getBatchSize(),
                                 std::numeric_limits<LABEL_T>::max()));
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

  // #pragma omp parallel
  {
    std::vector<uint32_t> buffer(_largest_doc);

    // #pragma omp for
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

template <typename LABEL_T>
MaxFlashArray<LABEL_T>::~MaxFlashArray() {
  for (uint32_t i = 0; i < _maxflash_array.size(); i++) {
    delete _maxflash_array.at(i);
  }
  delete _function;
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
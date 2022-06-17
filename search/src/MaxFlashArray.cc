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
      // This goes up to _hash_function->numTables() inclusive because it is
      // possible for a point to collide anywhere in [0, num_tables]
      _collision_count_to_sim(_hash_function->numTables() + 1) {
  for (uint32_t collision_count = 0;
       collision_count <= _collision_count_to_sim.size(); collision_count++) {
    float table_collision_probability =
        static_cast<float>(collision_count) / _hash_function->numTables();
    // For a given query and datapoint, if their similariy is sim, then each of
    // the hashes_per_table hash functions in the LSH table has collision
    // probability sim. Since the point collides only if all hashes_per_table
    // hashes collide, we have that
    // sim^hashes_per_table = table_collision_probability
    // Simplifying,
    // sim = e^(ln(table_collion_probability) / hashes_per_table)
    _collision_count_to_sim[collision_count] =
        std::exp(std::log(table_collision_probability) / hashes_per_table);
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
      // We normalize different size queries (thereby computing avg sim instead
      // of max sim) in order to not overweight long queries. This only matter
      // when different queries can have different number of embeddings.
      result[i] /= query.getBatchSize();
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
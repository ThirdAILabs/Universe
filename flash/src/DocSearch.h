#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include "MaxFlashArray.h"
#include "Utils.h"
#include <hashing/src/FastSRP.h>
#include <dataset/src/Vectors.h>
#include <dataset/src/batch_types/DenseBatch.h>
#include <exceptions/src/Exceptions.h>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::search {

// TODO(josh): This class is NOT currently safe to call concurrently.
// TODO(josh): Right now this only has support for dense input and documents
// with a max of 256 embeddings
/**
 * Represents a service that allows document addition, removal, and queries.
 * For now, can represent at most 2^32 - 1 documents.
 */
class DocSearch {
 public:
  DocSearch(uint32_t hashes_per_table, uint32_t num_tables, uint32_t dense_dim,
            const std::vector<std::vector<float>>& centroids_input)
      : _dense_dim(dense_dim),
        _nprobe_query(2),
        _largest_internal_id(0),
        _num_centroids(centroids_input.size()),
        _document_array(new thirdai::hashing::FastSRP(
                            dense_dim, hashes_per_table, num_tables),
                        hashes_per_table),
        _centroids(dense_dim * centroids_input.size()),
        _centroid_id_to_internal_id(centroids_input.size()) {
    for (uint32_t i = 0; i < centroids_input.size(); i++) {
      if (_dense_dim != centroids_input.at(i).size()) {
        throw std::invalid_argument(
            "Every centroids must have dimension equal to dense_dim. Instead"
            " found centroid " +
            std::to_string(i) + " with dimension of " +
            std::to_string(centroids_input.at(i).size()) +
            " and passed in dense_dim of " + std::to_string(_dense_dim));
      }
      // We copy the centroids because I couldn't get pybind keep_alive to work.
      for (uint32_t d = 0; d < dense_dim; d++) {
        _centroids.at(i * dense_dim + d) = centroids_input.at(i).at(d);
      }
    }
  }

  // Returns true if this is a new document, false if this was an old document
  // and we updated it.
  bool addDocument(const dataset::DenseBatch& embeddings,
                   const std::string& doc_id, const std::string& doc_text) {
    std::vector<uint32_t> centroid_ids = getNearestCentroids(embeddings, 1);
    return addDocument(embeddings, centroid_ids, doc_id, doc_text);
  }

  // Returns true if this is a new document, false if this was an old document
  // and we updated it.
  bool addDocument(const dataset::DenseBatch& embeddings,
                   const std::vector<uint32_t>& centroid_ids,
                   const std::string& doc_id, const std::string& doc_text) {
    bool deletedOldDoc = deleteDocument(doc_id);

    uint32_t internal_id = _document_array.addDocument(embeddings);
    _largest_internal_id = std::max(_largest_internal_id, internal_id);

    for (uint32_t centroid_id : centroid_ids) {
      _centroid_id_to_internal_id.at(centroid_id).push_back(internal_id);
    }

    _doc_id_to_doc_text[doc_id] = doc_text;

    _doc_id_to_internal_id[doc_id] = internal_id;

    _internal_id_to_doc_id.resize(internal_id + 1);
    _internal_id_to_doc_id.at(internal_id) = doc_id;

    return deletedOldDoc;
  }

  bool deleteDocument(const std::string& doc_id) {
    if (!_doc_id_to_doc_text.count(doc_id)) {
      return false;
    }

    // TODO(josh)
    throw thirdai::exceptions::NotImplemented(
        "Deleting documents is not yet implemented.");
  }

  std::optional<std::string> getDocument(const std::string& doc_id) const {
    if (!_doc_id_to_doc_text.count(doc_id)) {
      return {};
    }
    return _doc_id_to_doc_text.at(doc_id);
  }

  std::vector<std::pair<std::string, std::string>> query(
      const dataset::DenseBatch& embeddings, uint32_t top_k) const {
    if (embeddings.getBatchSize() == 0) {
      throw std::invalid_argument("Need at least one query vector but found 0");
    }
    for (uint32_t i = 0; i < embeddings.getBatchSize(); i++) {
      if (embeddings.at(i).dim() != _dense_dim) {
        throw std::invalid_argument("Vector " + std::to_string(i) +
                                    " has dimension " + std::to_string(i) +
                                    " but should have dimension equal to the "
                                    "original passed in dense dimension, " +
                                    std::to_string(_dense_dim));
      }
    }

    std::vector<uint32_t> centroid_ids =
        getNearestCentroids(embeddings, _nprobe_query);
    std::vector<uint32_t> top_k_internal_ids =
        frequencyCountCentroidBuckets(centroid_ids, top_k);
    std::vector<uint32_t> reranked =
        rankDocuments(embeddings, top_k_internal_ids);
    std::vector<std::pair<std::string, std::string>> result;
    for (uint32_t id : reranked) {
      std::string doc_id = _internal_id_to_doc_id.at(id);
      result.emplace_back(doc_id, _doc_id_to_doc_text.at(doc_id));
    }
    return result;
  }

  // Delete copy constructor and assignment
  DocSearch(const DocSearch&) = delete;
  DocSearch& operator=(const DocSearch&) = delete;

 protected:
  // This needs to be protected since it's a top level serialization target
  // called by a child class, but DO NOT call it unless you are creating a
  // temporary object to serialize into.
  DocSearch(){};

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_dense_dim, _nprobe_query, _largest_internal_id, _num_centroids,
            _document_array, _centroids, _centroid_id_to_internal_id,
            _doc_id_to_doc_text, _doc_id_to_internal_id,
            _internal_id_to_doc_id);
  }

  uint32_t _dense_dim, _nprobe_query, _largest_internal_id, _num_centroids;
  MaxFlashArray<uint8_t> _document_array;
  std::vector<float> _centroids;
  std::vector<std::vector<uint32_t>> _centroid_id_to_internal_id;
  std::unordered_map<std::string, std::string> _doc_id_to_doc_text;
  std::unordered_map<std::string, uint32_t> _doc_id_to_internal_id;
  std::vector<std::string> _internal_id_to_doc_id;

  // Finds the nearest nprobe centroids for each vector in the batch and
  // then concatenates all of the centroid ids across the batch.
  std::vector<uint32_t> getNearestCentroids(const dataset::DenseBatch& batch,
                                            uint32_t nprobe) const {
    std::vector<uint32_t> result;
#pragma omp parallel for default(none) shared(batch, nprobe, result)
    for (uint32_t i = 0; i < batch.getBatchSize(); i++) {
      std::vector<uint32_t> nearest_centroids =
          getNearestCentroids(batch.at(i), nprobe);
      for (uint32_t probe : getNearestCentroids(batch.at(i), nprobe)) {
        (void)probe;
#pragma omp critical
        result.push_back(probe);
      }
    }

    removeDuplicates(result);
    return result;
  }

  // Finds the nearest nprobe centroids for the single vector passed in
  std::vector<uint32_t> getNearestCentroids(const dataset::DenseVector& vec,
                                            uint32_t nprobe) const {
    std::vector<float> scores(_num_centroids);
    for (uint32_t i = 0; i < _num_centroids; i++) {
      float dot = 0;
      for (uint32_t d = 0; d < _dense_dim; d++) {
        dot += _centroids.at(i * _dense_dim + d) * vec.at(d);
      }
      scores.at(i) = dot;
    }
    return argmax(scores, nprobe);
  }

  std::vector<uint32_t> frequencyCountCentroidBuckets(
      const std::vector<uint32_t>& centroid_ids, uint32_t top_k) const {
    // Populate initial counts array.
    std::vector<int32_t> counts(_largest_internal_id + 1, 0);
    for (uint32_t centroid_id : centroid_ids) {
      for (uint32_t internal_id : _centroid_id_to_internal_id.at(centroid_id)) {
        counts.at(internal_id) += 1;
      }
    }

    // Find the top k by looping through the input again to know what counts
    // values to examine (we can avoid traversing all possible counts, many
    // of which are zero). Since we do it this way for performance we can't
    // use the argmax util function. We negate a counts element when we have
    // seen it so that if we come across it again (if there is more than one
    // occurence of it) we can ignore it.
    // Note also that the heap is a max heap, so we negate everything to make
    // it a min heap in effect.

    std::priority_queue<std::pair<int32_t, uint32_t>> heap;
    for (uint32_t centroid_id : centroid_ids) {
      for (uint32_t internal_id : _centroid_id_to_internal_id.at(centroid_id)) {
        if (counts.at(internal_id) < 0) {
          continue;
        }
        counts.at(internal_id) *= -1;
        int32_t negative_count = counts.at(internal_id);
        if (heap.size() < top_k || negative_count < heap.top().first) {
          heap.emplace(negative_count, internal_id);
        }
        if (heap.size() > top_k) {
          heap.pop();
        }
      }
    }

    std::vector<uint32_t> result;
    while (!heap.empty()) {
      result.push_back(heap.top().second);
      heap.pop();
    }

    std::reverse(result.begin(), result.end());
    return result;
  }

  // This method returns a permutation of the input internal_ids_to_rerank
  // sorted in descending order by the approximated score of that document.
  std::vector<uint32_t> rankDocuments(
      const dataset::DenseBatch& query_embeddings,
      const std::vector<uint32_t>& internal_ids_to_rerank) const {
    std::vector<float> document_scores = _document_array.getDocumentScores(
        query_embeddings, internal_ids_to_rerank);

    // This is a little confusing, these are indices into the
    // internal_ids_to_rerank array. We then need to convert them back to
    // document internal_ids, which we do below.
    std::vector<uint32_t> sorted_indices = argsort_descending(document_scores);

    std::vector<uint32_t> permuted_ids(internal_ids_to_rerank.size());
    for (uint32_t i = 0; i < permuted_ids.size(); i++) {
      permuted_ids[i] = internal_ids_to_rerank.at(sorted_indices[i]);
    }

    return permuted_ids;
  }
};

}  // namespace thirdai::search
#pragma once

#include <wrappers/src/EigenDenseWrapper.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include "MaxFlashArray.h"
#include "Utils.h"
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/FastSRP.h>
#include <Eigen/src/Core/util/Constants.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/CheckLicense.h>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::search {

//
/**
 * Represents a service that allows document addition, removal, and queries.
 * For now, can represent at most 2^32 - 1 documents. Right now this only has
 * support for documents with a max of 256 embeddings. If there are more than
 * 256 embeddings, it silently truncates. This class is NOT currently safe to
 * call concurrently.
 */
class DocSearch {
 public:
  DocSearch(uint32_t hashes_per_table, uint32_t num_tables, uint32_t dense_dim,
            const std::vector<std::vector<float>>& centroids_input)
      : _dense_dim(dense_dim),
        _nprobe_query(2),
        _largest_internal_id(0),
        _num_centroids(centroids_input.size()),
        _centroids(dense_dim, centroids_input.size()),
        _centroid_id_to_internal_id(centroids_input.size()) {
    thirdai::licensing::checkLicense();

    if (dense_dim == 0 || num_tables == 0 || hashes_per_table == 0) {
      throw std::invalid_argument(
          "The passed in dense dimension, number of tables, and hashes per "
          "table must all be greater than 0.");
    }
    if (centroids_input.empty()) {
      throw std::invalid_argument(
          "Must pass in at least one centroid, found 0.");
    }
    _nprobe_query = std::min<uint64_t>(centroids_input.size(), _nprobe_query);
    for (uint32_t i = 0; i < centroids_input.size(); i++) {
      if (_dense_dim != centroids_input.at(i).size()) {
        throw std::invalid_argument(
            "Every centroids must have dimension equal to dense_dim. Instead"
            " found centroid " +
            std::to_string(i) + " with dimension of " +
            std::to_string(centroids_input.at(i).size()) +
            " and passed in dense_dim of " + std::to_string(_dense_dim));
      }

      // We delay constructing this so that we can do sanitize input in
      // this constructor rather than in FastSRP and MaxFlashArray
      _document_array = std::make_unique<MaxFlashArray<uint8_t>>(
          new thirdai::hashing::FastSRP(dense_dim, hashes_per_table,
                                        num_tables),
          hashes_per_table);
    }

    for (uint32_t centroid_id = 0; centroid_id < centroids_input.size();
         centroid_id++) {
      if (_dense_dim != centroids_input.at(centroid_id).size()) {
        throw std::invalid_argument(
            "Every centroids must have dimension equal to dense_dim. Instead"
            " found centroid " +
            std::to_string(centroid_id) + " with dimension of " +
            std::to_string(centroids_input.at(centroid_id).size()) +
            " and passed in dense_dim of " + std::to_string(_dense_dim));
      }
      for (uint32_t d = 0; d < dense_dim; d++) {
        // Note we are populating the centroid matrix so that it is tranposed,
        // allowing us to avoid transposing during multiplication.
        _centroids(d, centroid_id) = centroids_input.at(centroid_id).at(d);
      }
    }
  }

  // Returns true if this is a new document, false if this was an old document
  // and we updated it.
  bool addDocument(const BoltBatch& embeddings, const std::string& doc_id,
                   const std::string& doc_text) {
    std::vector<uint32_t> centroid_ids = getNearestCentroids(embeddings, 1);
    return addDocumentWithCentroids(embeddings, centroid_ids, doc_id, doc_text);
  }

  // Returns true if this is a new document, false if this was an old document
  // and we updated it.
  bool addDocumentWithCentroids(const BoltBatch& embeddings,
                                const std::vector<uint32_t>& centroid_ids,
                                const std::string& doc_id,
                                const std::string& doc_text) {
    bool deletedOldDoc = deleteDocument(doc_id);

    // The document array assigned the new document to an "internal_id" when
    // we add it, which now becomes the integer representing this document in
    // our system. They differ from the passed in doc_id, which is an arbitrary
    // string that uniquely identifies the document; the internal_id is the
    // next available smallest positive integer that from now on uniquely
    // identifies the document.
    uint32_t internal_id = _document_array->addDocument(embeddings);
    _largest_internal_id = std::max(_largest_internal_id, internal_id);

    for (uint32_t centroid_id : centroid_ids) {
      _centroid_id_to_internal_id.at(centroid_id).push_back(internal_id);
    }

    _doc_id_to_doc_text[doc_id] = doc_text;

    _doc_id_to_internal_id[doc_id] = internal_id;

    // We need to call resize here instead of simply push_back because the
    // internal_id we get assigned might not necessarily be equal to the
    // push_back index, since if we concurrently add two documents at the same
    // time race conditions might reorder who gets assigned an id first and
    // who gets to this line first.
    if (internal_id >= _internal_id_to_doc_id.size()) {
      _internal_id_to_doc_id.resize(internal_id + 1);
    }
    _internal_id_to_doc_id.at(internal_id) = doc_id;

    return !deletedOldDoc;
  }

  bool deleteDocument(const std::string& doc_id) {
    if (!_doc_id_to_doc_text.count(doc_id)) {
      return false;
    }
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
      const BoltBatch& embeddings, uint32_t top_k,
      uint32_t num_to_rerank) const {
    std::vector<uint32_t> centroid_ids =
        getNearestCentroids(embeddings, _nprobe_query);
    return queryWithCentroids(embeddings, centroid_ids, top_k, num_to_rerank);
  }

  std::vector<std::pair<std::string, std::string>> queryWithCentroids(
      const BoltBatch& embeddings, const std::vector<uint32_t>& centroid_ids,
      uint32_t top_k, uint32_t num_to_rerank) const {
    if (embeddings.getBatchSize() == 0) {
      throw std::invalid_argument("Need at least one query vector but found 0");
    }
    if (top_k == 0) {
      throw std::invalid_argument(
          "The passed in top_k must be at least 1, was 0");
    }
    if (top_k > num_to_rerank) {
      throw std::invalid_argument(
          "The passed in top_k must be <= the passed in num_to_rerank");
    }
    for (uint32_t i = 0; i < embeddings.getBatchSize(); i++) {
      if (embeddings[i].len != _dense_dim) {
        throw std::invalid_argument("Embedding " + std::to_string(i) +
                                    " has dimension " +
                                    std::to_string(embeddings[i].len) +
                                    " but should have dimension equal to the "
                                    "original passed in dense dimension, " +
                                    std::to_string(_dense_dim));
      }
      if (!embeddings[i].isDense()) {
        throw std::invalid_argument("Embedding " + std::to_string(i) +
                                    " is sparse but should be dense");
      }
    }

    std::vector<uint32_t> top_k_internal_ids =
        frequencyCountCentroidBuckets(centroid_ids, num_to_rerank);

    std::vector<uint32_t> reranked =
        rankDocuments(embeddings, top_k_internal_ids);

    std::vector<std::pair<std::string, std::string>> result;
    for (uint32_t i = 0; i < std::min<uint32_t>(reranked.size(), top_k); i++) {
      uint32_t id = reranked.at(i);
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
  DocSearch() { thirdai::licensing::checkLicense(); };

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
  // This is a uinque_ptr rather than the object itself so that we can delay
  // constructing it until after input sanitization; see the constructor for m
  // more information.
  std::unique_ptr<MaxFlashArray<uint8_t>> _document_array;
  // These are stored tranposed for ease of multiplication
  Eigen::MatrixXf _centroids;
  std::vector<std::vector<uint32_t>> _centroid_id_to_internal_id;
  std::unordered_map<std::string, std::string> _doc_id_to_doc_text;
  std::unordered_map<std::string, uint32_t> _doc_id_to_internal_id;
  std::vector<std::string> _internal_id_to_doc_id;

  // Finds the nearest nprobe centroids for each vector in the batch and
  // then concatenates all of the centroid ids across the batch.
  std::vector<uint32_t> getNearestCentroids(const BoltBatch& batch,
                                            uint32_t nprobe) const {
    Eigen::MatrixXf eigen_batch(batch.getBatchSize(), _dense_dim);
    for (uint32_t i = 0; i < batch.getBatchSize(); i++) {
      for (uint32_t d = 0; d < _dense_dim; d++) {
        eigen_batch(i, d) = batch[i].activations[d];
      }
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        eigen_result = eigen_batch * _centroids;
    std::vector<uint32_t> nearest_centroids(batch.getBatchSize() * nprobe);

#pragma omp parallel for default(none) \
    shared(batch, eigen_result, nprobe, nearest_centroids, Eigen::Dynamic)
    for (uint32_t i = 0; i < batch.getBatchSize(); i++) {
      std::vector<uint32_t> probe_results = argmax(eigen_result.row(i), nprobe);
      for (uint32_t p = 0; p < nprobe; p++) {
        nearest_centroids.at(i * nprobe + p) = probe_results.at(p);
      }
    }

    removeDuplicates(nearest_centroids);
    return nearest_centroids;
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
      // Top is the pair with the smallest score still in the heap as the first
      // element and the internal_id as the second element, so we push back
      // the second element, the internal_id, into the result vector.
      result.push_back(heap.top().second);
      heap.pop();
    }

    std::reverse(result.begin(), result.end());
    return result;
  }

  // This method returns a permutation of the input internal_ids_to_rerank
  // sorted in descending order by the approximated score of that document.
  std::vector<uint32_t> rankDocuments(
      const BoltBatch& query_embeddings,
      const std::vector<uint32_t>& internal_ids_to_rerank) const {
    // This returns a vector of scores, where the ith score is the score of
    // the document with the internal_id at internal_ids_to_rerank[i]
    std::vector<float> document_scores = _document_array->getDocumentScores(
        query_embeddings, internal_ids_to_rerank);

    // This is a little confusing, these sorted_indices are indices into
    // the document_scores array and represent a ranking of the
    // internal_ids_to_rerank. We still need to use this ranking to permute and
    // return internal_ids_to_rerank, which we do below.
    std::vector<uint32_t> sorted_indices = argsort_descending(document_scores);

    std::vector<uint32_t> permuted_ids(internal_ids_to_rerank.size());
    for (uint32_t i = 0; i < permuted_ids.size(); i++) {
      permuted_ids[i] = internal_ids_to_rerank.at(sorted_indices[i]);
    }

    return permuted_ids;
  }
};

}  // namespace thirdai::search
#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include "MaxFlashArray.h"
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
        _document_array(new thirdai::hashing::FastSRP(
                            dense_dim, hashes_per_table, num_tables),
                        hashes_per_table),
        _centroid_id_to_internal_id(centroids_input.size()) {
    // We copy the centroids here because I couldn't get pybind keep_alive to
    // work.
    for (const std::vector<float>& c : centroids_input) {
      _centroids.push_back(c);
    }
    for (uint32_t i = 0; i < _centroids.size(); i++) {
      if (_dense_dim != _centroids.at(i).size()) {
        throw std::invalid_argument(
            "Every centroids must have dimension equal to dense_dim. Instead"
            " found centroid " +
            std::to_string(i) + " with dimension of " +
            std::to_string(_centroids.at(i).size()) +
            " and passed in dense_dim of " + std::to_string(_dense_dim));
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
    throw thirdai::exceptions::NotImplemented();
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
    archive(_dense_dim, _nprobe_query, _largest_internal_id, _document_array,
            _centroids, _centroid_id_to_internal_id, _doc_id_to_doc_text,
            _doc_id_to_internal_id, _internal_id_to_doc_id);
  }

  uint32_t _dense_dim, _nprobe_query, _largest_internal_id;
  MaxFlashArray<uint8_t> _document_array;
  std::vector<std::vector<float>> _centroids;
  std::vector<std::vector<uint32_t>> _centroid_id_to_internal_id;
  std::unordered_map<std::string, std::string> _doc_id_to_doc_text;
  std::unordered_map<std::string, uint32_t> _doc_id_to_internal_id;
  std::vector<std::string> _internal_id_to_doc_id;

  // Finds the nearest nprobe centroids for each vector in the batch and
  // then concatenates all of the centroid ids across the batch.
  std::vector<uint32_t> getNearestCentroids(const dataset::DenseBatch& batch,
                                            uint32_t nprobe) const {
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < batch.getBatchSize(); i++) {
      for (uint32_t probe : getNearestCentroids(batch.at(i), nprobe)) {
        result.push_back(probe);
      }
    }

    removeDuplicates(result);
    return result;
  }

  // Finds the nearest nprobe centroids for the single vector passed in
  std::vector<uint32_t> getNearestCentroids(const dataset::DenseVector& vec,
                                            uint32_t nprobe) const {
    std::vector<float> scores(_centroids.size());
#pragma omp parallel for default(none) shared(vec, scores)
    for (uint32_t i = 0; i < _centroids.size(); i++) {
      float dot = 0;
      for (uint32_t j = 0; j < _dense_dim; j++) {
        dot += _centroids.at(i).at(j) * vec.at(j);
      }
      scores.at(i) = dot;
    }
    return argmax(scores, nprobe);
  }

  static void removeDuplicates(std::vector<uint32_t>& v) {
    std::sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
  }

  // TODO(josh): Move some of these static functions to different file
  template <class T>
  static std::vector<uint32_t> argmax(const std::vector<T>& input,
                                      uint32_t top_k) {
    static_assert(std::is_signed<T>::value,
                  "The input to the argmax needs to be signed so we can negate "
                  "the values for the priority queue.");
    std::priority_queue<std::pair<T, uint32_t>> min_heap;
    for (uint32_t i = 0; i < input.size(); i++) {
      if (min_heap.size() < top_k) {
        min_heap.emplace(-input[i], i);
      } else if (-input[i] < min_heap.top().first) {
        min_heap.pop();
        min_heap.emplace(-input[i], i);
      }
    }

    std::vector<uint32_t> result;
    while (!min_heap.empty()) {
      result.push_back(min_heap.top().second);
      min_heap.pop();
    }

    std::reverse(result.begin(), result.end());

    return result;
  }

  std::vector<uint32_t> frequencyCountCentroidBuckets(
      const std::vector<uint32_t>& centroid_ids, uint32_t top_k) const {
    // TODO(josh): This can be made much more efficient
    std::vector<int32_t> counts(_largest_internal_id + 1, 0);
    for (uint32_t centroid_id : centroid_ids) {
      for (uint32_t internal_id : _centroid_id_to_internal_id.at(centroid_id)) {
        counts.at(internal_id) += 1;
      }
    }
    return argmax(counts, top_k);
  }

  std::vector<uint32_t> rankDocuments(
      const dataset::DenseBatch& query_embeddings,
      const std::vector<uint32_t>& documents_to_rerank) const {
    std::vector<float> document_scores = _document_array.getDocumentScores(
        query_embeddings, documents_to_rerank);

    // Get indices of top elements in sim_sums
    std::vector<uint32_t> idx(document_scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&document_scores](size_t i1, size_t i2) {
                       return document_scores[i1] > document_scores[i2];
                     });

    assert(documents_to_rerank.size() == idx.size());
    std::vector<uint32_t> final_result(documents_to_rerank.size());
    for (uint32_t i = 0; i < final_result.size(); i++) {
      final_result[i] = documents_to_rerank.at(idx[i]);
    }

    return final_result;
  }
};

}  // namespace thirdai::search
#pragma once

#include <cereal/archives/binary.hpp>
#include <hashing/src/FastSRP.h>
#include <hashing/src/HashFunction.h>
#include <dataset/src/NumpyDataset.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <search/src/DocSearch.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>

namespace py = pybind11;

namespace thirdai::search::python {

void createSearchSubmodule(py::module_& module);

// TODO(josh): Make uint8_t configurable, will currently cut off all documents
// at 256 embeddings
class PyDocSearch final : public DocSearch {
 public:
  PyDocSearch(const std::vector<std::vector<float>>& centroids,
              uint32_t hashes_per_table, uint32_t num_tables,
              uint32_t dense_dim)
      : DocSearch(hashes_per_table, num_tables, dense_dim, centroids) {}

  bool addDocument(
      const std::string& doc_id, const std::string& doc_text,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          embeddings) {
    auto single_batch_dataset = dataset::numpy::numpyToBoltVectorDataset(
        embeddings, /* batch_size = */ std::numeric_limits<uint64_t>::max());
    return DocSearch::addDocument(single_batch_dataset->at(0), doc_id,
                                  doc_text);
  }

  bool addDocumentWithCentroids(
      const std::string& doc_id, const std::string& doc_text,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          embeddings,
      const std::vector<uint32_t>& doc_centroid_ids) {
    auto single_batch_dataset = dataset::numpy::numpyToBoltVectorDataset(
        embeddings, /* batch_size = */ std::numeric_limits<uint64_t>::max());
    return DocSearch::addDocumentWithCentroids(
        single_batch_dataset->at(0), doc_centroid_ids, doc_id, doc_text);
  }

  bool deleteDocument(const std::string& doc_id) {
    return DocSearch::deleteDocument(doc_id);
  }

  std::optional<std::string> getDocument(const std::string& doc_id) {
    return DocSearch::getDocument(doc_id);
  }

  py::array query(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          embeddings,
      uint32_t top_k, uint32_t num_to_rerank) {
    auto single_batch_dataset = dataset::numpy::numpyToBoltVectorDataset(
        embeddings, /* batch_size = */ std::numeric_limits<uint64_t>::max());
    std::vector<std::pair<std::string, std::string>> result =
        DocSearch::query(single_batch_dataset->at(0), top_k, num_to_rerank);
    return py::cast(std::move(result));
  }

  py::array queryWithCentroids(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          embeddings,
      const std::vector<uint32_t>& query_centroid_ids, uint32_t top_k,
      uint32_t num_to_rerank) {
    auto single_batch_dataset = dataset::numpy::numpyToBoltVectorDataset(
        embeddings, /* batch_size = */ std::numeric_limits<uint64_t>::max());
    std::vector<std::pair<std::string, std::string>> result =
        DocSearch::queryWithCentroids(single_batch_dataset->at(0),
                                      query_centroid_ids, top_k, num_to_rerank);
    return py::cast(std::move(result));
  }

  void serialize_to_file(const std::string& path) {
    std::ofstream filestream(path, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<PyDocSearch> deserialize_from_file(
      const std::string& path) {
    std::ifstream filestream(path, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<PyDocSearch> serialize_into(new PyDocSearch());
    iarchive(*serialize_into);
    return serialize_into;
  }

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& ar) {
    // See https://uscilab.github.io/cereal/inheritance.html
    ar(cereal::base_class<DocSearch>(this));
  }
  // Private constructor to construct an empty object for Cereal. See
  // https://uscilab.github.io/cereal/
  PyDocSearch() : DocSearch() {}
};

}  // namespace thirdai::search::python
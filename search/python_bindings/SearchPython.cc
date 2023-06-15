#include "BeamSearch.h"
#include "DocSearchPython.h"
#include <pybind11/buffer_info.h>
#include <pybind11/stl.h>
#include <search/src/HNSW.h>
#include <memory>
#include <stdexcept>

namespace thirdai::search::python {

using NumpyArray =
    py::array_t<float, py::array::forcecast | py::array::c_style>;

class PyHNSW {
 public:
  PyHNSW(size_t max_nbrs, NumpyArray data, size_t construction_buffer_size,
         size_t num_initializations = 100)
      : _data(std::move(data)) {
    if (_data.ndim() != 2) {
      throw std::invalid_argument("Expected data to be 2D numpy array.");
    }

    size_t n_nodes = _data.shape(0);
    size_t dim = _data.shape(1);

    _index = std::make_unique<hnsw::HNSW>(max_nbrs, dim, n_nodes, _data.data(),
                                          construction_buffer_size,
                                          num_initializations);
  }

  std::vector<uint32_t> query(const NumpyArray& query, size_t k,
                              size_t search_buffer_size,
                              size_t num_initializations = 100) {
    if (query.ndim() != 1 || query.shape(0) != _data.shape(1)) {
      throw std::invalid_argument("Expected query to have shape (" +
                                  std::to_string(_data.shape(1)) + ",)");
    }

    return _index->query(query.data(), k, search_buffer_size,
                         num_initializations);
  }

 private:
  std::unique_ptr<hnsw::HNSW> _index;
  NumpyArray _data;
};

void createSearchSubmodule(py::module_& module) {
  auto search_submodule = module.def_submodule("search");
  py::class_<PyDocSearch>(
      search_submodule, "DocRetrieval",
      "The DocRetrieval module allows you to build, query, save, and load a "
      "semantic document search index.")
      .def(py::init<const std::vector<std::vector<float>>&, uint32_t, uint32_t,
                    uint32_t>(),
           py::arg("centroids"), py::arg("hashes_per_table"),
           py::arg("num_tables"), py::arg("dense_input_dimension"),
           "Constructs a new DocRetrieval index. Centroids should be a "
           "two-dimensional array of floats, where each row is of length "
           "dense_input_dimension (the dimension of the document embeddings). "
           "hashes_per_table and num_tables are hyperparameters for the doc "
           "sketches. Roughly, increase num_tables to increase accuracy at the"
           "cost of speed and memory (you can try powers of 2; a good starting "
           "value is 32). Hashes_per_table should be around log_2 the average"
           "document size (by number of embeddings).")
      .def("add_doc", &PyDocSearch::addDocument, py::arg("doc_id"),
           py::arg("doc_text"), py::arg("doc_embeddings"),
           "Adds a new document to the DocRetrieval index. If the doc_id "
           "already exists in the index, this will overwrite it. The "
           "doc_embeddings should be a two dimensional numpy array of the "
           "document's embeddings. Each row should be of length "
           "dense_input_dimension. doc_text is only needed if you want it to "
           "be retrieved in calls to get_doc and query. Returns true if this"
           "was a new document and false otherwise.")
      .def("add_doc", &PyDocSearch::addDocumentWithCentroids, py::arg("doc_id"),
           py::arg("doc_text"), py::arg("doc_embeddings"),
           py::arg("doc_centroid_ids"),
           "A normal add, except also accepts the ids of the closest centroid"
           "to each of the doc_embeddings if these are "
           "precomputed (helpful for batch adds).")
      .def("delete_doc", &PyDocSearch::deleteDocument, py::arg("doc_id"),
           "Delete the document with the passed doc_id if such a document "
           "exists, otherwise this is a NOOP. Returns true if the document "
           "was succesfully deleted, false if no document with doc_id was "
           "found.")
      .def("get_doc", &PyDocSearch::getDocument, py::arg("doc_id"),
           "Returns the doc_text of the document with doc_id, or None if no "
           "document with doc_id was found.")
      .def(
          "query", &PyDocSearch::query, py::arg("query_embeddings"),
          py::arg("top_k"), py::arg("num_to_rerank") = 8192,
          "Finds the best top_k documents that are most likely to semantically "
          "answer the query. There is an additional optional parameter here "
          "called num_to_rerank that represents how many documents you want "
          "us to "
          "internally rerank. The default of 8192 is fine for most use cases.")
      .def("query", &PyDocSearch::queryWithCentroids,
           py::arg("query_embeddings"), py::arg("query_centroid_ids"),
           py::arg("top_k"), py::arg("num_to_rerank") = 8192,
           "A normal query, except also accepts the ids of the closest centroid"
           "to each of the query_embeddings")
      .def("serialize_to_file", &PyDocSearch::serialize_to_file,
           py::arg("output_path"),
           "Serialize the DocRetrieval index to a file.")
      .def_static("deserialize_from_file", &PyDocSearch::deserialize_from_file,
                  py::arg("input_path"),
                  "Deserialize the DocRetrieval index from a file.");

  search_submodule.def("beam_search", &beamSearchBatch,
                       py::arg("probabilities"), py::arg("transition_matrix"),
                       py::arg("beam_size"));

  py::class_<PyHNSW>(search_submodule, "HNSW")
      .def(py::init<size_t, NumpyArray, size_t, size_t>(), py::arg("max_nbrs"),
           py::arg("data"), py::arg("construction_buffer_size"),
           py::arg("num_initializations") = 100)
      .def("query", &PyHNSW::query, py::arg("query"), py::arg("k"),
           py::arg("search_buffer_size"), py::arg("num_initializations") = 100);
}

}  // namespace thirdai::search::python
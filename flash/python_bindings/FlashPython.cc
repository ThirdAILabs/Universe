#include "FlashPython.h"
#include "DocSearchPython.h"
#include <pybind11/stl.h>

namespace thirdai::search::python {

void createSearchSubmodule(py::module_& module) {
  auto search_submodule = module.def_submodule("search");
  py::class_<PyFlash>(
      search_submodule, "MagSearch",
      "MagSearch is an index for performing near neighbor search. To use it, "
      "construct an index by passing in a hash function and then calling "
      "add() at least once to populate the index.")
      .def(py::init<hashing::HashFunction&, uint32_t>(),
           py::arg("hash_function"), py::arg("reservoir_size"),
           "Builds a MagSearch index where all hash "
           "buckets have a max size reservoir_size.")
      .def(py::init<hashing::HashFunction&>(), py::arg("hash_function"),
           "Builds a MagSearch index where buckets do not have a max size.")
      .def("add", &PyFlash::addDenseBatch, py::arg("dense_data"),
           py::arg("starting_index"),
           "Adds a dense numpy batch to the "
           "index, where each row represents a vector with sequential ids "
           "starting from the passed in starting_index.")
      .def("add", &PyFlash::addSparseBatch, py::arg("sparse_values"),
           py::arg("sparse_indices"), py::arg("starting_index"),
           "Adds a sparse batch batch to the "
           "index, where each corresponding pair of items from sparse_values "
           "and sparse_indices represents a sparse vector. The vectors have "
           "sequential ids starting from the passed in starting_index.")
      .def("query", &PyFlash::queryDenseBatch, py::arg("dense_queries"),
           py::arg("top_k") = 10,
           "Performs a batch query that returns the "
           "approximate top_k neighbors as a row for each of the passed in "
           "queries.")
      .def("query", &PyFlash::querySparseBatch, py::arg("sparse_query_values"),
           py::arg("sparse_query_indices"), py::arg("top_k") = 10,
           "Performs a batch query that returns the "
           "approximate top_k neighbors as a row for each of the passed in "
           "queries.");

  // TODO(josh): Right now this only has support for dense input and documents
  // with a max of 256 embeddings, and can not be parallilized
  py::class_<PyDocSearch>(search_submodule, "doc_retrieval_index")
      .def(py::init<const std::vector<std::vector<float>>&, uint32_t, uint32_t,
                    uint32_t>(),
           py::arg("centroids"), py::arg("hashes_per_table"),
           py::arg("num_tables"), py::arg("dense_input_dimension"))
      .def("add_document", &PyDocSearch::addDocumentWithCentroids,
           py::arg("doc_id"), py::arg("doc_text"), py::arg("doc_embeddings"),
           py::arg("centroid_ids"))
      .def("add_document", &PyDocSearch::addDocument, py::arg("doc_id"),
           py::arg("doc_text"), py::arg("doc_embeddings"))
      .def("delete_document", &PyDocSearch::deleteDocument, py::arg("doc_id"))
      .def("get_document", &PyDocSearch::getDocument, py::arg("doc_id"))
      .def("query", &PyDocSearch::query, py::arg("query_embeddings"),
           py::arg("top_k"))
      .def("serialize_to_file", &PyDocSearch::serialize_to_file,
           py::arg("output_path"))
      .def_static("deserialize_from_file", &PyDocSearch::deserialize_from_file,
                  py::arg("input_path"));
}

}  // namespace thirdai::search::python
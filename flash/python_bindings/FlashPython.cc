#include "FlashPython.h"
#include "DocSearchPython.h"

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

  // TODO(josh): Right now this only has support for dense input
  py::class_<PyMaxFlashArray>(search_submodule, "doc_retrieval_index")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("dense_input_dimension"),
           py::arg("max_allowable_doc_size") = 256)
      .def("add_document", &PyMaxFlashArray::addDocument,
           py::arg("document_embeddings"))
      .def("rank_documents", &PyMaxFlashArray::rankDocuments,
           py::arg("query_embeddings"), py::arg("document_ids_to_rank"))
      .def("serialize_to_file", &PyMaxFlashArray::serialize_to_file,
           py::arg("output_path"))
      .def_static("deserialize_from_file",
                  &PyMaxFlashArray::deserialize_from_file,
                  py::arg("input_path"));
}

}  // namespace thirdai::search::python
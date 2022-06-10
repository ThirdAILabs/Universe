#include "BloltPython.h"
#include "DocSearchPython.h"
#include "FlashPython.h"
#include <pybind11/stl.h>

namespace thirdai::search::python {

void createSearchSubmodule(py::module_& module) {
  auto search_submodule = module.def_submodule("search");
#if THIRDAI_EXPOSE_ALL
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
#endif

  // TODO(josh): Right now this only has support for dense input and documents
  // with a max of 256 embeddings, and can not be parallilized
  // TODO(josh): Comment this class more
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

#if THIRDAI_EXPOSE_ALL
  py::class_<PyBlolt>(search_submodule, "BoltSearch")
      .def(py::init<uint64_t, uint8_t, uint64_t>(),
           py::arg("estimated_dataset_size"), py::arg("num_classifiers"),
           py::arg("input_dim"))
      .def("index", &PyBlolt::index, py::arg("train_data"), py::arg("all_data"),
           py::arg("batch_size"))
      .def("query", &PyBlolt::query, py::arg("query_batch_python"),
           py::arg("top_k"))
      .def("serialize_to_file", &PyBlolt::serialize_to_file,
           py::arg("output_path"), "Serialize the Blolt index to a file.")
      .def_static("deserialize_from_file", &PyBlolt::deserialize_from_file,
                  py::arg("input_path"),
                  "Deserialize the Blolt index from a file.");

#endif
}

}  // namespace thirdai::search::python
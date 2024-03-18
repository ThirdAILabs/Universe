#include "BeamSearch.h"
#include "DocSearchPython.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>
#include <search/src/InvertedIndex.h>

namespace thirdai::search::python {

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

  py::class_<InvertedIndex, std::shared_ptr<InvertedIndex>>(search_submodule,
                                                            "InvertedIndex")
      .def(py::init<float, float, float, bool, bool>(),
           py::arg("idf_cutoff_frac") = InvertedIndex::DEFAULT_IDF_CUTOFF_FRAC,
           py::arg("k1") = InvertedIndex::DEFAULT_K1,
           py::arg("b") = InvertedIndex::DEFAULT_B, py::arg("stem") = true,
           py::arg("lowercase") = true)
      .def("index", &InvertedIndex::index, py::arg("ids"), py::arg("docs"))
      .def("query", &InvertedIndex::queryBatch, py::arg("queries"),
           py::arg("k"))
      .def("query", &InvertedIndex::query, py::arg("query"), py::arg("k"))
      .def("remove", &InvertedIndex::remove, py::arg("ids"))
      .def_static("parallel_query", &InvertedIndex::parallelQuery,
                  py::arg("indices"), py::arg("query"), py::arg("k"))
      .def("save", &InvertedIndex::save, py::arg("filename"))
      .def_static("load", &InvertedIndex::load, py::arg("filename"))
      .def(bolt::python::getPickleFunction<InvertedIndex>());
}

}  // namespace thirdai::search::python
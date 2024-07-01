#include "BeamSearch.h"
#include "DocSearchPython.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>
#include <search/src/inverted_index/FinetunableRetriever.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/OnDiskIndex.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <optional>

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

  py::class_<Tokenizer, TokenizerPtr>(search_submodule, "Tokenizer");  // NOLINT

  py::class_<DefaultTokenizer, Tokenizer, std::shared_ptr<DefaultTokenizer>>(
      search_submodule, "DefaultTokenizer")
      .def(py::init<bool, bool>(), py::arg("stem") = true,
           py::arg("lowercase") = true);

  py::class_<WordKGrams, Tokenizer, std::shared_ptr<WordKGrams>>(
      search_submodule, "WordKGrams")
      .def(py::init<uint32_t, bool, bool, bool, bool>(), py::arg("k") = 4,
           py::arg("soft_start") = true, py::arg("include_whole_words") = true,
           py::arg("stem") = true, py::arg("lowercase") = true)
      .def("tokenize", &WordKGrams::tokenize, py::arg("input"));

  py::class_<IndexConfig>(search_submodule, "IndexConfig");  // NOLINT

  py::class_<InvertedIndex, std::shared_ptr<InvertedIndex>>(search_submodule,
                                                            "InvertedIndex")
      .def(py::init<const IndexConfig&>(), py::arg("config") = IndexConfig())
      .def("index", &InvertedIndex::index, py::arg("ids"), py::arg("docs"))
      .def("query", &InvertedIndex::queryBatch, py::arg("queries"),
           py::arg("k"))
      .def("query", &InvertedIndex::query, py::arg("query"), py::arg("k"),
           py::arg("parallelize") = true)
      .def("rank", &InvertedIndex::rankBatch, py::arg("queries"),
           py::arg("candidates"), py::arg("k"))
      .def("rank", &InvertedIndex::rank, py::arg("query"),
           py::arg("candidates"), py::arg("k"), py::arg("parallelize") = true)
      .def("remove", &InvertedIndex::remove, py::arg("ids"))
      .def("size", &InvertedIndex::size)
      .def("update_idf_cutoff", &InvertedIndex::updateIdfCutoff,
           py::arg("cutoff"))
      .def("save", &InvertedIndex::save, py::arg("filename"))
      .def_static("load", &InvertedIndex::load, py::arg("filename"))
      .def(py::pickle(
          /**
           * This is to achive compatability between neuraldb's with indexes
           * that were pickled using cereal vs the archives. This try/catch
           * logic is done here instead of in load_stream so that the binary
           * stream can be reset before attempting to load with cereal if
           * loading with the archive fails.
           */
          [](const std::shared_ptr<InvertedIndex>& index) -> py::bytes {
            std::stringstream ss;
            index->save_stream(ss);
            return py::bytes(ss.str());
          },
          [](const py::bytes& binary_index) -> std::shared_ptr<InvertedIndex> {
            py::buffer_info info(py::buffer(binary_index).request());
            char* data_ptr = reinterpret_cast<char*>(info.ptr);

            try {
              bolt::python::Membuf sbuf(data_ptr, data_ptr + info.size);
              std::istream input(&sbuf);
              return InvertedIndex::load_stream(input);
            } catch (...) {
              bolt::python::Membuf sbuf(data_ptr, data_ptr + info.size);
              std::istream input(&sbuf);
              return InvertedIndex::load_stream_cereal(input);
            }
          }));

  py::class_<OnDiskIndex, std::shared_ptr<OnDiskIndex>>(search_submodule,
                                                        "OnDiskIndex")
      .def(py::init<const std::string&>(), py::arg("db_name"))
      .def("index", &OnDiskIndex::index, py::arg("ids"), py::arg("docs"))
      .def("query", &OnDiskIndex::query, py::arg("query"), py::arg("k"),
           py::arg("parallelize") = false)
      .def("prune", &OnDiskIndex::prune)
      .def("save", &OnDiskIndex::save, py::arg("filename"))
      .def_static("load", &OnDiskIndex::load, py::arg("filename"));

  py::class_<FinetunableRetriever, std::shared_ptr<FinetunableRetriever>>(
      search_submodule, "FinetunableRetriever")
      .def(py::init<float, uint32_t, uint32_t, const IndexConfig&,
                    const std::optional<std::string>&>(),
           py::arg("lambda") = FinetunableRetriever::DEFAULT_LAMBDA,
           py::arg("min_top_docs") = FinetunableRetriever::DEFAULT_MIN_TOP_DOCS,
           py::arg("top_queries") = FinetunableRetriever::DEFAULT_TOP_QUERIES,
           py::arg("config") = IndexConfig(),
           py::arg("save_path") = std::nullopt)
      .def("index", &FinetunableRetriever::index, py::arg("ids"),
           py::arg("docs"))
      .def("finetune", &FinetunableRetriever::finetune, py::arg("doc_ids"),
           py::arg("queries"))
      .def("associate", &FinetunableRetriever::associate, py::arg("sources"),
           py::arg("targets"), py::arg("strength") = 4)
      .def("query", &FinetunableRetriever::queryBatch, py::arg("queries"),
           py::arg("k"))
      .def("query", &FinetunableRetriever::query, py::arg("query"),
           py::arg("k"), py::arg("parallelize") = false)
      .def("rank", &FinetunableRetriever::rankBatch, py::arg("queries"),
           py::arg("candidates"), py::arg("k"))
      .def("rank", &FinetunableRetriever::rank, py::arg("query"),
           py::arg("candidates"), py::arg("k"), py::arg("parallelize") = false)
      .def("size", &FinetunableRetriever::size)
      .def("remove", &FinetunableRetriever::remove, py::arg("ids"))
      .def_static("train_from", &FinetunableRetriever::trainFrom,
                  py::arg("index"))
      .def("save", &FinetunableRetriever::save, py::arg("filename"))
      .def_static("load", &FinetunableRetriever::load, py::arg("filename"));
}

}  // namespace thirdai::search::python
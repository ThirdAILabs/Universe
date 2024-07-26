#pragma once

#include <search/src/inverted_index/id_map/IdMap.h>
#include <search/src/inverted_index/IndexConfig.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/Retriever.h>
#include <utils/UUID.h>
#include <unordered_map>

namespace thirdai::search {

using QueryId = uint64_t;

class FinetunableRetriever {
 public:
  static constexpr float DEFAULT_LAMBDA = 0.6;
  static constexpr uint32_t DEFAULT_MIN_TOP_DOCS = 20;
  static constexpr uint32_t DEFAULT_TOP_QUERIES = 10;

  explicit FinetunableRetriever(const IndexConfig& config,
                                const std::optional<std::string>& save_path)
      : FinetunableRetriever(DEFAULT_LAMBDA, DEFAULT_MIN_TOP_DOCS,
                             DEFAULT_TOP_QUERIES, config, save_path) {}

  explicit FinetunableRetriever(
      float lambda = DEFAULT_LAMBDA,
      uint32_t min_top_docs = DEFAULT_MIN_TOP_DOCS,
      uint32_t top_queries = DEFAULT_TOP_QUERIES,
      const IndexConfig& config = IndexConfig(),
      const std::optional<std::string>& save_path = std::nullopt);

  static std::shared_ptr<FinetunableRetriever> trainFrom(
      const std::shared_ptr<InvertedIndex>& index) {
    auto retriever = std::make_shared<FinetunableRetriever>();
    retriever->_doc_index = index;
    return retriever;
  }

  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs);

  void finetune(const std::vector<std::vector<DocId>>& doc_ids,
                const std::vector<std::string>& queries);

  void associate(const std::vector<std::string>& sources,
                 const std::vector<std::string>& targets, uint32_t strength);

  std::vector<DocScore> query(const std::string& query, uint32_t k,
                              bool parallelize = true) const;

  std::vector<std::vector<DocScore>> queryBatch(
      const std::vector<std::string>& queries, uint32_t k) const;

  std::vector<DocScore> rank(const std::string& query,
                             const std::unordered_set<DocId>& candidates,
                             uint32_t k, bool parallelize = true) const;

  std::vector<std::vector<DocScore>> rankBatch(
      const std::vector<std::string>& queries,
      const std::vector<std::unordered_set<DocId>>& candidates,
      uint32_t k) const;

  void prune() { _doc_index->prune(); }

  void remove(const std::vector<DocId>& ids);

  size_t size() const { return _doc_index->size(); }

  void save(const std::string& save_path) const;

  static std::shared_ptr<FinetunableRetriever> load(
      const std::string& save_path, bool read_only);

  // This is deprecated, it is only for compatability loading old models since
  // we need a load/save stream method to define the pickle function.
  void save_stream(std::ostream& ostream) const;

  // This is deprecated, it is only for compatability loading old models.
  static std::shared_ptr<FinetunableRetriever> load_stream(
      std::istream& istream);

 private:
  explicit FinetunableRetriever(const std::string& save_path, bool read_only);

  ar::ConstArchivePtr metadataToArchive() const;

  void metadataFromArchive(const ar::Archive& archive);

  // This is deprecated, it is only for compatability loading old models.
  explicit FinetunableRetriever(const ar::Archive& archive);

  std::shared_ptr<Retriever> _doc_index;
  std::shared_ptr<Retriever> _query_index;

  std::unique_ptr<IdMap> _query_to_docs;

  float _lambda;
  uint32_t _min_top_docs;
  uint32_t _top_queries;

  utils::uuid::UUIDGenerator _uuid_gen;
};

}  // namespace thirdai::search
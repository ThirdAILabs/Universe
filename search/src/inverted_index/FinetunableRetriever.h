#pragma once

#include <search/src/inverted_index/InvertedIndex.h>
#include <unordered_map>

namespace thirdai::search {

using QueryId = uint64_t;

class FinetunableRetriever {
 public:
  static constexpr float DEFAULT_LAMBDA = 0.6;
  static constexpr uint32_t DEFAULT_MIN_TOP_DOCS = 20;
  static constexpr uint32_t DEFAULT_TOP_QUERIES = 10;

  explicit FinetunableRetriever(
      float lambda = DEFAULT_LAMBDA,
      uint32_t min_top_docs = DEFAULT_MIN_TOP_DOCS,
      uint32_t top_queries = DEFAULT_TOP_QUERIES,
      size_t shard_size = InvertedIndex::DEFAULT_SHARD_SIZE);

  explicit FinetunableRetriever(const ar::Archive& archive);

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

  std::vector<DocScore> query(const std::string& query, uint32_t k) const;

  std::vector<std::vector<DocScore>> queryBatch(
      const std::vector<std::string>& queries, uint32_t k) const;

  std::vector<DocScore> rank(const std::string& query,
                             const std::unordered_set<DocId>& candidates,
                             uint32_t k) const;

  std::vector<std::vector<DocScore>> rankBatch(
      const std::vector<std::string>& queries,
      const std::vector<std::unordered_set<DocId>>& candidates,
      uint32_t k) const;

  void remove(const std::vector<DocId>& ids);

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<FinetunableRetriever> fromArchive(
      const ar::Archive& archive);

  void save(const std::string& filename) const;

  void save_stream(std::ostream& ostream) const;

  size_t size() const { return _doc_index->size(); }

  std::shared_ptr<InvertedIndex> docIndex() { return _doc_index; }

  static std::shared_ptr<FinetunableRetriever> load(
      const std::string& filename);

  static std::shared_ptr<FinetunableRetriever> load_stream(
      std::istream& istream);

 private:
  std::shared_ptr<InvertedIndex> _doc_index;
  std::shared_ptr<InvertedIndex> _query_index;

  std::unordered_map<QueryId, std::vector<DocId>> _query_to_docs;
  std::unordered_map<DocId, std::vector<QueryId>> _doc_to_queries;

  QueryId _next_query_id = 0;

  float _lambda;
  uint32_t _min_top_docs;
  uint32_t _top_queries;
};

}  // namespace thirdai::search
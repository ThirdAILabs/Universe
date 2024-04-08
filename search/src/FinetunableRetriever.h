#pragma once

#include <search/src/InvertedIndex.h>
#include <unordered_map>

namespace thirdai::search {

using QueryId = uint64_t;

class FinetunableRetriever {
 public:
  static constexpr float DEFAULT_LAMBDA = 0.6;
  static constexpr uint32_t DEFAULT_MIN_TOP_DOCS = 20;
  static constexpr uint32_t DEFAULT_TOP_QUERIES = 20;

  explicit FinetunableRetriever(float lambda = DEFAULT_LAMBDA,
                                uint32_t min_top_docs = DEFAULT_MIN_TOP_DOCS,
                                uint32_t top_queries = DEFAULT_TOP_QUERIES);

  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs);

  void finetune(const std::vector<std::vector<DocId>>& doc_ids,
                const std::vector<std::string>& queries);

  std::vector<DocScore> query(const std::string& query, uint32_t k) const;

  std::vector<std::vector<DocScore>> queryBatch(
      const std::vector<std::string>& queries, uint32_t k) const;

 private:
  std::shared_ptr<InvertedIndex> _doc_index;
  std::shared_ptr<InvertedIndex> _query_index;
  std::unordered_map<QueryId, std::vector<DocId>> _query_to_docs;

  QueryId _queries_indexed = 0;

  float _lambda;
  uint32_t _min_top_docs;
  uint32_t _top_queries;
};

}  // namespace thirdai::search
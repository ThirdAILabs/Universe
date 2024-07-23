#pragma once

#include <rocksdb/db.h>
#include <rocksdb/utilities/transaction_db.h>
#include <search/src/inverted_index/DbAdapter.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/Retriever.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <unordered_map>

namespace thirdai::search {

using HashedToken = uint32_t;

class OnDiskIndex final : public Retriever {
 public:
  explicit OnDiskIndex(const std::string& save_path,
                       const IndexConfig& config = IndexConfig());

  OnDiskIndex(const std::string& save_path, std::unique_ptr<DbAdapter> db,
              const IndexConfig& config);

  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs) final;

  void update(const std::vector<DocId>& ids,
              const std::vector<std::string>& extra_tokens) final;

  std::vector<DocScore> query(const std::string& query, uint32_t k,
                              bool parallelize) const final;

  std::vector<DocScore> rank(const std::string& query,
                             const std::unordered_set<DocId>& candidates,
                             uint32_t k, bool parallelize) const final;

  void remove(const std::vector<DocId>& ids) final;

  size_t size() const final { return _db->getNDocs(); }

  void save(const std::string& new_save_path) const final;

  static std::shared_ptr<OnDiskIndex> load(const std::string& save_path,
                                           bool read_only);

  std::string type() const final { return typeName(); }

  static std::string typeName() { return "on-disk"; }

  void prune() final;

 private:
  std::pair<std::vector<uint32_t>,
            std::vector<std::unordered_map<HashedToken, uint32_t>>>
  countTokenOccurences(const std::vector<std::string>& docs) const;

  std::unordered_map<DocId, float> scoreDocuments(
      const std::string& query) const;

  std::vector<HashedToken> tokenize(const std::string& text) const;

  std::unique_ptr<DbAdapter> _db;

  std::string _save_path;

  // Query variables
  uint64_t _max_docs_to_score;
  float _max_token_occurrence_frac;
  float _k1;
  float _b;

  TokenizerPtr _tokenizer;
};

}  // namespace thirdai::search
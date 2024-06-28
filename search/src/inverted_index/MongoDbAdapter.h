#pragma once

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/database.hpp>
#include <mongocxx/collection.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <exception>
#include <search/src/inverted_index/DbAdapter.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>

namespace thirdai::search {

class MongoDbAdapter final : public DbAdapter {
 public:
  explicit MongoDbAdapter(const std::string& db_uri, const std::string& db_name);

  void storeDocLens(const std::vector<DocId>& ids, const std::vector<uint32_t>& doc_lens) final;
  void updateTokenToDocs(const std::unordered_map<HashedToken, std::vector<DocCount>>& token_to_new_docs) final;
  std::vector<SerializedDocCountIterator> lookupDocs(const std::vector<HashedToken>& query_tokens) const final;
  uint32_t getDocLen(DocId doc_id) const final;
  uint64_t getNDocs() const final;
  uint64_t getSumDocLens() const final;

 private:
  mongocxx::client _client;
  mongocxx::database _db;
  mongocxx::collection _docs;
  mongocxx::collection _tokens;
};

}  // namespace thirdai::search

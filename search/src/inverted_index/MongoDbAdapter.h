#pragma once

#include <mongocxx/client.hpp>
#include <mongocxx/collection.hpp>
#include <mongocxx/database.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <search/src/inverted_index/DbAdapter.h>
#include <cstdint>
#include <exception>
#include <unordered_map>
#include <vector>

namespace thirdai::search {

class MongoDbAdapter final : public DbAdapter {
 public:
  // TODO(pratik): This may throw error creating multiple OnDiskIndesk.
  mongocxx::instance global_instance{};

  explicit MongoDbAdapter(const std::string& db_uri, const std::string& db_name,
                          uint32_t bulk_update_batch_size = 64000);

  void updateSumDocLens(int64_t additional_len);
  void storeDocLens(const std::vector<DocId>& ids,
                    const std::vector<uint32_t>& doc_lens) final;
  void updateTokenToDocs(
      const std::unordered_map<HashedToken, std::vector<DocCount>>&
          token_to_new_docs) final;
  std::vector<SerializedDocCountIterator> lookupDocs(
      const std::vector<HashedToken>& query_tokens) final;
  uint32_t getDocLen(DocId doc_id) final;
  uint64_t getNDocs() final;
  uint64_t getSumDocLens() final;

  std::string createFormattedLogLine(const std::string& operation,
                                     size_t batchSize, long long duration);

 private:
  mongocxx::client _client;
  mongocxx::database _db;
  mongocxx::collection _docs;
  mongocxx::collection _tokens;

  uint32_t _bulk_update_batch_size;
};

}  // namespace thirdai::search

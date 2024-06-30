#pragma once

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/database.hpp>
#include <mongocxx/collection.hpp>
#include <search/src/inverted_index/DbAdapter.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <exception>

namespace thirdai::search {

class MongoDbAdapter final : public DbAdapter {


public:

    mongocxx::instance global_instance{};
    
    explicit MongoDbAdapter(const std::string& db_uri, const std::string& db_name);

    void storeDocLens(const std::vector<DocId>& ids, const std::vector<uint32_t>& doc_lens) final;
    void updateTokenToDocs(const std::unordered_map<HashedToken, std::vector<DocCount>>& token_to_new_docs) final;
    std::vector<SerializedDocCountIterator> lookupDocs(const std::vector<HashedToken>& query_tokens) final;
    uint32_t getDocLen(DocId doc_id) final;
    uint64_t getNDocs() final;
    uint64_t getSumDocLens() final;

private:
    mongocxx::client _client;
    mongocxx::database _db;
    mongocxx::collection _docs;
    mongocxx::collection _tokens;
};

}  // namespace thirdai::search

#include "MongoDbAdapter.h"

namespace thirdai::search {

MongoDbAdapter::MongoDbAdapter(const std::string& db_uri, const std::string& db_name) {
    mongocxx::instance inst{};
    mongocxx::uri uri(db_uri);
    _client = mongocxx::client(uri);


    _db = _client[db_name];
    _docs = _db["docs"];
    _tokens = _db["tokens"];
}

void MongoDbAdapter::storeDocLens(const std::vector<DocId>& ids, const std::vector<uint32_t>& doc_lens) {
    if (ids.size() != doc_lens.size()) {
      throw std::invalid_argument("IDs and document lengths must match in size.");
    }
    for (size_t i = 0; i < ids.size(); ++i) {
      auto builder = bsoncxx::builder::stream::document{};
      bsoncxx::document::value doc_value = builder
        << "doc_id" << static_cast<int64_t>(ids[i])
        << "doc_len" << static_cast<int32_t>(doc_lens[i])
        << bsoncxx::builder::stream::finalize;
      _docs.insert_one(doc_value.view());
    }
}

void MongoDbAdapter::updateTokenToDocs(const std::unordered_map<HashedToken, std::vector<DocCount>>& token_to_new_docs) {
    for (const auto& pair : token_to_new_docs) {
      for (const auto& doc_count : pair.second) {
        auto builder = bsoncxx::builder::stream::document{};
        bsoncxx::document::value update_doc = builder
          << "$push" << bsoncxx::builder::stream::open_document
          << "docs" << bsoncxx::builder::stream::open_document
          << "doc_id" << static_cast<int64_t>(doc_count.doc_id)
          << "count" << static_cast<int32_t>(doc_count.count)
          << bsoncxx::builder::stream::close_document
          << bsoncxx::builder::stream::close_document
          << bsoncxx::builder::stream::finalize;

        _tokens.update_one(
          builder << "token" << static_cast<int32_t>(pair.first) << bsoncxx::builder::stream::finalize,
          update_doc.view());
      }
    }
}

std::vector<SerializedDocCountIterator> MongoDbAdapter::lookupDocs(const std::vector<HashedToken>& query_tokens) const {
    std::vector<SerializedDocCountIterator> results;
    for (auto token : query_tokens) {
        auto cursor = _tokens.find(bsoncxx::builder::stream::document{} << "token" << static_cast<int32_t>(token) << bsoncxx::builder::stream::finalize);
        std::string serialized;
        for (auto&& doc : cursor) {
            auto docs = doc["docs"].get_array().value;
            for (auto&& d : docs) {
                DocCount dc(d["doc_id"].get_int64().value, d["count"].get_int32().value);
                serialized.append(reinterpret_cast<const char*>(&dc), sizeof(DocCount));
            }
        }
        results.emplace_back(std::move(serialized));
    }
    return results;
}

uint32_t MongoDbAdapter::getDocLen(DocId doc_id) const {
    auto result = _docs.find_one(bsoncxx::builder::stream::document{} << "doc_id" << static_cast<int64_t>(doc_id) << bsoncxx::builder::stream::finalize);
    if (result) {
      return result->view()["doc_len"].get_int32().value;
    }
    throw std::runtime_error("Document length not found.");
}

uint64_t MongoDbAdapter::getNDocs() const {
    return _docs.count_documents({});
}

uint64_t MongoDbAdapter::getSumDocLens() const {
    mongocxx::pipeline p{};
    p.group(bsoncxx::builder::stream::document{} << "_id" << nullptr << "total" << bsoncxx::builder::stream::open_document << "$sum" << "$doc_len" << bsoncxx::builder::stream::close_document << bsoncxx::builder::stream::finalize);
    auto cursor = _docs.aggregate(p);
    if (auto doc = cursor.begin(); doc != cursor.end()) {
        return (*doc)["total"].get_int64().value;
    }
    return 0;
}

}  // namespace thirdai::search

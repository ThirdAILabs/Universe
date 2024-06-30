#include "MongoDbAdapter.h"
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/json.hpp>

using bsoncxx::builder::basic::document;
using bsoncxx::builder::basic::kvp;
using bsoncxx::builder::basic::make_document;
using bsoncxx::types::b_null;


namespace thirdai::search {

MongoDbAdapter::MongoDbAdapter(const std::string& db_uri, const std::string& db_name) {
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
        document builder{};
        builder.append(kvp("doc_id", static_cast<int64_t>(ids[i])), 
                       kvp("doc_len", static_cast<int32_t>(doc_lens[i])));
        _docs.insert_one(builder.extract());
    }
}

void MongoDbAdapter::updateTokenToDocs(const std::unordered_map<HashedToken, std::vector<DocCount>>& token_to_new_docs) {
    for (const auto& pair : token_to_new_docs) {
        for (const auto& doc_count : pair.second) {
            document builder{};
            builder.append(kvp("token", static_cast<int64_t>(pair.first)));
            document update_doc{};
            update_doc.append(
                kvp("$push", make_document(
                    kvp("docs", make_document(
                        kvp("doc_id", static_cast<int64_t>(doc_count.doc_id)),
                        kvp("count", static_cast<int32_t>(doc_count.count))
                    ))
                ))
            );
            _tokens.update_one(builder.extract(), update_doc.extract());
        }
    }
}

std::vector<SerializedDocCountIterator> MongoDbAdapter::lookupDocs(const std::vector<HashedToken>& query_tokens) {
    std::vector<SerializedDocCountIterator> results;
    for (auto token : query_tokens) {
        auto cursor = _tokens.find(make_document(kvp("token", bsoncxx::types::b_int64{static_cast<int64_t>(token)})));
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

uint32_t MongoDbAdapter::getDocLen(DocId doc_id) {
    auto result = _docs.find_one(make_document(kvp("doc_id", bsoncxx::types::b_int64{static_cast<int64_t>(doc_id)})));
    if (result) {
        return result->view()["doc_len"].get_int32().value;
    }
    throw std::runtime_error("Document length not found.");
}

uint64_t MongoDbAdapter::getNDocs() {
    return _docs.count_documents(make_document());
}

uint64_t MongoDbAdapter::getSumDocLens() {
    mongocxx::pipeline p{};
    p.group(make_document(
        kvp("_id", b_null{}),
        kvp("total", make_document(
            kvp("$sum", "$doc_len")
        ))
    ));
    auto cursor = _docs.aggregate(p);
    if (auto doc = cursor.begin(); doc != cursor.end()) {
        return (*doc)["total"].get_int32().value;
    }
    return 0;
}

}  // namespace thirdai::search

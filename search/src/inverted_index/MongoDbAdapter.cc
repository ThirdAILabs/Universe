#include "MongoDbAdapter.h"
#include <bolt/src/utils/ProgressBar.h>
#include <bolt/src/utils/Timer.h>
#include <bsoncxx/builder/basic/document.hpp>
#include <bsoncxx/json.hpp>
#include <utils/Logging.h>
#include <sstream>
#include <stdexcept>

using bsoncxx::builder::basic::document;
using bsoncxx::builder::basic::kvp;
using bsoncxx::builder::basic::make_document;
using bsoncxx::types::b_null;
using mongocxx::model::update_one;

namespace thirdai::search {

MongoDbAdapter::MongoDbAdapter(const std::string& db_uri,
                               const std::string& db_name,
                               uint32_t bulk_update_batch_size) {
  mongocxx::uri uri(db_uri);
  _client = mongocxx::client(uri);
  _db = _client[db_name];
  _docs = _db["docs"];
  _docs.create_index(make_document(kvp("doc_id", 1)),
                     mongocxx::options::index{}.unique(true));

  // for storing sum
  _docs.insert_one(make_document(kvp("doc_type", "metadata"), kvp("sum_doc_lens", 0)));

  _tokens = _db["tokens"];
  _tokens.create_index(make_document(kvp("token", 1)),
                       mongocxx::options::index{}.unique(true));

  _bulk_update_batch_size = bulk_update_batch_size;
}


void MongoDbAdapter::updateSumDocLens(int64_t additional_len) {
    _docs.update_one(
        make_document(kvp("doc_type", "metadata")),
        make_document(kvp("$inc", make_document(kvp("sum_doc_lens", additional_len))))
    );
}


void MongoDbAdapter::storeDocLens(const std::vector<DocId>& ids,
                                  const std::vector<uint32_t>& doc_lens) {
  if (ids.size() != doc_lens.size()) {
    throw std::invalid_argument("IDs and document lengths must match in size.");
  }

  int64_t sum_added_lens = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    document builder{};
    builder.append(kvp("doc_id", static_cast<int64_t>(ids[i])),
                   kvp("doc_len", static_cast<int32_t>(doc_lens[i])));
    _docs.insert_one(builder.extract());
    sum_added_lens += doc_lens[i];
  }

  updateSumDocLens(sum_added_lens);
}

std::string MongoDbAdapter::createFormattedLogLine(const std::string& operation,
                                                   size_t batchSize,
                                                   long long duration) {
  std::ostringstream log_stream;
  log_stream << operation << " - Batch Size: " << batchSize
             << ", Duration: " << duration << "ms";
  return log_stream.str();
}

void MongoDbAdapter::updateTokenToDocs(
    const std::unordered_map<HashedToken, std::vector<DocCount>>& token_to_new_docs) {
  auto bar = ProgressBar::makeOptional(true, "train", token_to_new_docs.size());
  std::unordered_map<HashedToken, bsoncxx::builder::basic::array> token_updates;
  size_t docs_processed = 0;

  bolt::utils::Timer batch_timer;

  // Lambda Function to process and append updates
  auto processUpdates = [&](auto& bulk, auto& updates) {
    for (auto& [token, docs_list] : updates) {
      bsoncxx::builder::basic::document builder{};
      builder.append(kvp("token", static_cast<int64_t>(token)));

      bsoncxx::builder::basic::document update_doc{};
      update_doc.append(
          kvp("$push",
              make_document(kvp("docs", make_document(kvp("$each", docs_list.extract()))))));

      mongocxx::model::update_one upsert_op{builder.extract(), update_doc.extract()};
      upsert_op.upsert(true);
      bulk.append(upsert_op);
    }
  };

  for (const auto& pair : token_to_new_docs) {
    for (const auto& doc_count : pair.second) {
      bsoncxx::document::value doc_entry =
          make_document(kvp("doc_id", static_cast<int64_t>(doc_count.doc_id)),
                        kvp("count", static_cast<int32_t>(doc_count.count)));
      token_updates[pair.first].append(std::move(doc_entry));
      docs_processed++;

      if (docs_processed >= _bulk_update_batch_size) {
        mongocxx::bulk_write bulk = _tokens.create_bulk_write();
        processUpdates(bulk, token_updates);
        auto result = bulk.execute();
        if (!result) {
          throw std::runtime_error("Error with bulk update!");
        }
        batch_timer.stop();
        std::string batch_time_log =
            createFormattedLogLine("Bulk doc training", docs_processed, batch_timer.milliseconds());
        logging::info(batch_time_log);
        batch_timer = bolt::utils::Timer();
        token_updates.clear();
        docs_processed = 0;
      }
    }

    if (bar) {
      bar->increment();
    }
  }

  if (!token_updates.empty()) {
    mongocxx::bulk_write bulk = _tokens.create_bulk_write();
    processUpdates(bulk, token_updates);
    auto result = bulk.execute();
    if (!result) {
      throw std::runtime_error("Error with final bulk update!");
    }
    batch_timer.stop();
    std::string batch_time_log = createFormattedLogLine(
        "Final bulk doc training", docs_processed, batch_timer.milliseconds());
    logging::info(batch_time_log);
  }
}


std::vector<SerializedDocCountIterator> MongoDbAdapter::lookupDocs(
    const std::vector<HashedToken>& query_tokens) {
  std::vector<SerializedDocCountIterator> results;
  bsoncxx::builder::basic::array array_builder;
  for (const auto& token : query_tokens) {
    array_builder.append(static_cast<int64_t>(token));
  }

  bolt::utils::Timer query_timer;
  auto query =
      bsoncxx::builder::basic::make_document(bsoncxx::builder::basic::kvp(
          "token",
          bsoncxx::builder::basic::make_document(
              bsoncxx::builder::basic::kvp("$in", array_builder.view()))));

  auto cursor = _tokens.find(query.view());

  std::unordered_map<int64_t, std::string> serialized_map;
  for (auto&& doc : cursor) {
    int64_t token = doc["token"].get_int64();
    std::string& serialized = serialized_map[token];
    auto docs = doc["docs"].get_array().value;
    for (auto&& d : docs) {
      DocCount dc(d["doc_id"].get_int64().value, d["count"].get_int32().value);
      serialized.append(reinterpret_cast<const char*>(&dc), sizeof(DocCount));
    }
  }

  for (const auto& token : query_tokens) {
    results.emplace_back(std::move(serialized_map[token]));
  }
  query_timer.stop();
  std::string query_time_log =
      createFormattedLogLine("Query Timing", 1, query_timer.milliseconds());
  logging::info(query_time_log);

  return results;
}

uint32_t MongoDbAdapter::getDocLen(DocId doc_id) {
  auto result = _docs.find_one(make_document(
      kvp("doc_id", bsoncxx::types::b_int64{static_cast<int64_t>(doc_id)})));
  if (result) {
    return result->view()["doc_len"].get_int32().value;
  }
  throw std::runtime_error("Document length not found.");
}

uint64_t MongoDbAdapter::getNDocs() {
  return _docs.count_documents(make_document());
}

uint64_t MongoDbAdapter::getSumDocLens() {
    auto result = _docs.find_one(make_document(kvp("doc_type", "metadata")));
    if (result) {
        return result->view()["sum_doc_lens"].get_int64().value;
    }
    throw std::runtime_error("Metadata for sum of document lengths not found.");
}

}  // namespace thirdai::search

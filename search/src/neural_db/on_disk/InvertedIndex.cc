#include "InvertedIndex.h"
#include <rocksdb/options.h>
#include <search/src/inverted_index/BM25.h>
#include <search/src/inverted_index/Utils.h>
#include <search/src/neural_db/on_disk/RocksDBError.h>

namespace thirdai::search::ndb {

const std::string N_CHUNKS = "n_chunks";
const std::string TOTAL_CHUNK_LEN = "total_chunk_len";
const std::string NEXT_CHUNK_ID = "next_chunk_id";

InvertedIndex::InvertedIndex(rocksdb::DB* db,
                             rocksdb::ColumnFamilyHandle* counters,
                             rocksdb::ColumnFamilyHandle* token_index,
                             const IndexConfig& config, bool read_only)
    : _db(db),
      _counters(counters),
      _token_index(token_index),
      _max_docs_to_score(config.max_docs_to_score),
      _max_token_occurrence_frac(config.max_token_occurrence_frac),
      _k1(config.k1),
      _b(config.b) {
  if (!read_only) {
    initCounter(N_CHUNKS, 0);
    initCounter(TOTAL_CHUNK_LEN, 0);
  }
}

void InvertedIndex::insert(TxnPtr& txn, ChunkId start_id,
                           const BatchTokens& token_counts,
                           const std::vector<uint32_t>& lens) {
  int64_t total_len_inc = 0;
  for (size_t i = 0; i < lens.size(); i++) {
    const ChunkId chunk_id = start_id + i;
    const auto chunk_key = asSlice<ChunkId>(&chunk_id);
    const int64_t chunk_len = lens[i];
    total_len_inc += chunk_len;

    auto status = txn->Put(_counters, chunk_key, asSlice<int64_t>(&chunk_len));
    if (!status.ok()) {
      throw RocksdbError(status, "inserting doc chunks");
    }
  }

  for (const auto& [token, chunk_counts] : token_counts) {
    const HashedToken token_value = token;
    const auto token_key = asSlice<HashedToken>(&token_value);
    const ChunkCountView view(chunk_counts);

    auto status = txn->Merge(_token_index, token_key, view.slice());
    if (!status.ok()) {
      throw RocksdbError(status, "inserting doc chunks");
    }
  }

  incrementCounter(txn, N_CHUNKS, lens.size());
  incrementCounter(txn, TOTAL_CHUNK_LEN, total_len_inc);
}

ChunkId InvertedIndex::reserveChunkIds(TxnPtr& txn, ChunkId n_ids) {
  std::string value;

  int64_t next_id;
  auto get_status = txn->GetForUpdate(rocksdb::ReadOptions(), _counters,
                                      NEXT_CHUNK_ID, &value);
  if (get_status.ok()) {
    if (value.size() != sizeof(int64_t)) {
      throw NeuralDbError(ErrorCode::MalformedData, "next id malformed");
    }
    next_id = *reinterpret_cast<int64_t*>(value.data());
  } else if (get_status.IsNotFound()) {
    next_id = 0;
  } else {
    throw RocksdbError(get_status, "retrieving next available chunk id");
  }

  int64_t new_next_id = next_id + n_ids;
  auto put_status =
      txn->Put(_counters, NEXT_CHUNK_ID, asSlice<int64_t>(&new_next_id));
  if (!put_status.ok()) {
    throw RocksdbError(put_status, "reserving chunk ids for doc");
  }

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "committing chunk id reservation");
  }

  return next_id;
}

std::unordered_map<ChunkId, float> InvertedIndex::candidateSet(
    const std::vector<HashedToken>& query_tokens) {
  const auto token_to_chunk_bytes = mapTokensToChunks(query_tokens);

  std::vector<ChunkCountView> token_to_chunks;
  token_to_chunks.reserve(token_to_chunk_bytes.size());
  for (const auto& chunks : token_to_chunk_bytes) {
    token_to_chunks.emplace_back(chunks);
  }

  const int64_t n_chunks = getCounter(N_CHUNKS);
  const int64_t total_chunk_len = getCounter(TOTAL_CHUNK_LEN);

  const auto token_to_idf = rankByIdf(token_to_chunks, n_chunks);

  const float avg_chunk_len = static_cast<float>(total_chunk_len) / n_chunks;

  std::unordered_map<ChunkId, float> chunk_scores;

  // This is used to cache the lens for docs that have already been seen to
  // avoid the DB lookup. This speeds up query processing.
  std::unordered_map<ChunkId, uint32_t> chunk_len_cache;

  const uint64_t query_len = query_tokens.size();

  for (const auto& [token_index, token_idf] : token_to_idf) {
    for (const auto& count : token_to_chunks[token_index]) {
      const ChunkId chunk_id = count.chunk_id;
      if (chunk_scores.size() < _max_docs_to_score ||
          chunk_scores.count(chunk_id)) {
        uint32_t chunk_len;
        if (chunk_len_cache.count(chunk_id)) {
          chunk_len = chunk_len_cache.at(chunk_id);
        } else {
          chunk_len = getCounter(asSlice<uint64_t>(&chunk_id));
          chunk_len_cache[chunk_id] = chunk_len;
        }

        const float score =
            bm25(/*idf=*/token_idf, /*cnt_in_doc=*/count.count,
                 /*doc_len=*/chunk_len, /*avg_doc_len=*/avg_chunk_len,
                 /*query_len=*/query_len, /*k1=*/_k1, /*b=*/_b);
        chunk_scores[chunk_id] += score;
      }
    }
  }

  return chunk_scores;
}

std::vector<rocksdb::PinnableSlice> InvertedIndex::mapTokensToChunks(
    const std::vector<HashedToken>& query_tokens) {
  std::vector<rocksdb::Slice> keys;
  keys.reserve(query_tokens.size());

  for (const HashedToken& token : query_tokens) {
    keys.emplace_back(asSlice<HashedToken>(&token));
  }
  std::vector<rocksdb::PinnableSlice> values(keys.size());
  std::vector<rocksdb::Status> statuses(keys.size());

  _db->MultiGet(rocksdb::ReadOptions(), _token_index, keys.size(), keys.data(),
                values.data(), statuses.data());

  std::vector<rocksdb::PinnableSlice> found;
  found.reserve(query_tokens.size());

  for (size_t i = 0; i < query_tokens.size(); i++) {
    if (statuses[i].IsNotFound()) {
      continue;
    }
    if (!statuses[i].ok()) {
      throw RocksdbError(statuses[i], "retrieving chunk data");
    }

    found.push_back(std::move(values[i]));
  }

  return found;
}

std::vector<std::pair<size_t, float>> InvertedIndex::rankByIdf(
    const std::vector<ChunkCountView>& token_to_chunks,
    int64_t n_chunks) const {
  const int64_t max_chunks_with_token =
      std::max<int64_t>(_max_token_occurrence_frac * n_chunks, 1000);

  std::vector<std::pair<size_t, float>> token_idx_to_idf;
  for (size_t i = 0; i < token_to_chunks.size(); i++) {
    if (!token_to_chunks[i].isPruned()) {
      const int64_t n_chunks_w_token = token_to_chunks[i].size();
      if (n_chunks_w_token < max_chunks_with_token) {
        token_idx_to_idf.emplace_back(i, idf(n_chunks, n_chunks_w_token));
      }
    }
  }

  std::sort(token_idx_to_idf.begin(), token_idx_to_idf.end(),
            HighestScore<size_t>{});

  return token_idx_to_idf;
}

void InvertedIndex::deleteChunks(TxnPtr& txn,
                                 const std::unordered_set<ChunkId>& chunk_ids) {
  auto iter = std::unique_ptr<rocksdb::Iterator>(
      txn->GetIterator(rocksdb::ReadOptions(), _token_index));

  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    ChunkCountView curr_value(iter->value());

    std::vector<ChunkCount> new_value;
    new_value.reserve(curr_value.size());

    for (const auto& count : curr_value) {
      if (chunk_ids.count(count.chunk_id)) {
        new_value.push_back(count);
      }
    }

    if (new_value.size() < curr_value.size()) {
      ChunkCountView view(new_value);
      auto put_status = txn->Put(_token_index, iter->key(), view.slice());
      if (!put_status.ok()) {
        throw RocksdbError(put_status, "removing doc chunks");
      }
    }
  }

  int64_t deleted_len = 0;

  for (const auto& chunk_id : chunk_ids) {
    const auto key = asSlice<ChunkId>(&chunk_id);
    std::string value;
    auto get_status =
        txn->GetForUpdate(rocksdb::ReadOptions(), _counters, key, &value);
    if (!get_status.ok()) {
      throw RocksdbError(get_status, "removing doc chunk data");
    }

    if (value.size() != sizeof(int64_t)) {
      throw NeuralDbError(ErrorCode::MalformedData, "counter is malformed");
    }

    deleted_len += *reinterpret_cast<int64_t*>(value.data());

    auto del_status = txn->Delete(_counters, key);
    if (!del_status.ok()) {
      throw RocksdbError(get_status, "removing doc chunk data");
    }
  }

  const int64_t n_chunks = chunk_ids.size();
  incrementCounter(txn, N_CHUNKS, -n_chunks);
  incrementCounter(txn, TOTAL_CHUNK_LEN, -deleted_len);
}

void InvertedIndex::prune(TxnPtr& txn) {
  const int64_t n_chunks = getCounter(N_CHUNKS);

  const size_t max_chunks_with_token =
      std::max<size_t>(_max_token_occurrence_frac * n_chunks, 1000);

  auto iter = std::unique_ptr<rocksdb::Iterator>(
      txn->GetIterator(rocksdb::ReadOptions(), _token_index));

  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    ChunkCountView chunks(iter->value());
    if (chunks.size() > max_chunks_with_token) {
      auto status =
          txn->Put(_token_index, iter->key(), asSlice<ChunkCount>(&PRUNED));
      if (!status.ok()) {
        throw RocksdbError(status, "pruning db");
      }
    }
  }

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "committing prune");
  }

  auto compact = _db->CompactRange(rocksdb::CompactRangeOptions(), _token_index,
                                   nullptr, nullptr);
  if (!compact.ok()) {
    throw RocksdbError(compact, "compacting db");
  }
}

void InvertedIndex::initCounter(const std::string& key, int64_t value) {
  auto status = _db->Put(rocksdb::WriteOptions(), _counters, key,
                         asSlice<int64_t>(&value));
  if (!status.ok()) {
    throw RocksdbError(status, "initializing counter");
  }
}

void InvertedIndex::incrementCounter(TxnPtr& txn, const std::string& key,
                                     int64_t value) {
  auto status = txn->Merge(_counters, key, asSlice<int64_t>(&value));
  if (!status.ok()) {
    throw RocksdbError(status, "incrementing counter");
  }
}

int64_t InvertedIndex::getCounter(const rocksdb::Slice& key) {
  std::string value;

  auto status = _db->Get(rocksdb::ReadOptions(), _counters, key, &value);
  if (!status.ok()) {
    throw RocksdbError(status, "retrieving counter");
  }

  if (value.size() != sizeof(int64_t)) {
    throw NeuralDbError(ErrorCode::MalformedData, "counter value malformed");
  }

  int64_t counter = *reinterpret_cast<int64_t*>(value.data());
  return counter;
}

}  // namespace thirdai::search::ndb
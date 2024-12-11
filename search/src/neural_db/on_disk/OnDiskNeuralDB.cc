#include "OnDiskNeuralDB.h"
#include <licensing/src/CheckLicense.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/status.h>
#include <search/src/inverted_index/BM25.h>
#include <search/src/inverted_index/Utils.h>
#include <search/src/neural_db/Constraints.h>
#include <search/src/neural_db/on_disk/MergeOperators.h>
#include <search/src/neural_db/on_disk/RocksDbError.h>
#include <search/src/neural_db/on_disk/Serialization.h>
#include <utils/UUID.h>
#include <array>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::search::ndb {

const std::string N_CHUNKS = "n_chunks";
const std::string TOTAL_CHUNK_LEN = "total_chunk_len";
const std::string NEXT_CHUNK_ID = "next_chunk_id";

std::string docVersionKey(const std::string& doc_id, uint32_t doc_version) {
  return doc_id + "_" + std::to_string(doc_version);
}

template <typename T>
inline rocksdb::Slice asSlice(const T* item) {
  return rocksdb::Slice(reinterpret_cast<const char*>(item), sizeof(T));
}

OnDiskNeuralDB::OnDiskNeuralDB(const std::string& save_path,
                               const IndexConfig& config, bool read_only)
    : _save_path(save_path),
      _max_docs_to_score(config.max_docs_to_score),
      _max_token_occurrence_frac(config.max_token_occurrence_frac),
      _k1(config.k1),
      _b(config.b),
      _text_processor(config.tokenizer) {
  licensing::checkLicense();

  createDirectory(save_path);

  rocksdb::Options options;
  options.create_if_missing = true;
  options.create_missing_column_families = true;

  rocksdb::ColumnFamilyOptions counter_options;
  counter_options.merge_operator = std::make_shared<IncrementCounter>();

  rocksdb::ColumnFamilyOptions concat_options;
  concat_options.merge_operator = std::make_shared<ConcatChunkCounts>();

  std::vector<rocksdb::ColumnFamilyDescriptor> column_families = {
      {rocksdb::kDefaultColumnFamilyName, {}},
      {"chunk_counters", counter_options},
      {"chunk_data", {}},
      {"chunk_metadata", {}},
      {"chunk_index", concat_options},
      {"doc_chunks", {}},
      {"doc_version", {}},
  };

  std::vector<rocksdb::ColumnFamilyHandle*> columns;

  rocksdb::Status open_status;
  if (!read_only) {
    open_status = rocksdb::TransactionDB::Open(
        options, rocksdb::TransactionDBOptions(), save_path, column_families,
        &columns, &_transaction_db);
    _db = _transaction_db;
  } else {
    open_status = rocksdb::DB::OpenForReadOnly(options, save_path,
                                               column_families, &columns, &_db);
    _transaction_db = nullptr;
  }
  if (!open_status.ok()) {
    throw RocksdbError(open_status, "unable to open database");
  }

  _default = columns.at(0);
  _chunk_counters = columns.at(1);
  _chunk_data = columns.at(2);
  _chunk_metadata = columns.at(3);
  _chunk_token_index = columns.at(4);
  _doc_chunks = columns.at(5);
  _doc_version = columns.at(6);

  if (!read_only) {
    auto txn = newTxn();
    incrementCounter(txn, N_CHUNKS, 0);
    incrementCounter(txn, TOTAL_CHUNK_LEN, 0);
    auto status = txn->Commit();
    if (!status.ok()) {
      throw RocksdbError(status, "initializing db");
    }
  }
}

void OnDiskNeuralDB::insert(const std::string& document,
                            const std::vector<std::string>& chunks,
                            const std::vector<MetadataMap>& metadata,
                            const std::optional<std::string>& doc_id_opt) {
  if (chunks.size() != metadata.size()) {
    throw std::invalid_argument("length of metadata and chunks must match");
  }

  const ChunkId start_id = reserveChunkIds(chunks.size());

  auto [token_counts, chunk_lens] = _text_processor.process(start_id, chunks);

  auto txn = newTxn();

  std::string doc_id;
  uint32_t doc_version;
  if (doc_id_opt) {
    doc_id = *doc_id_opt;
    doc_version = getDocVersion(txn, doc_id);
  } else {
    doc_id = utils::uuid::getRandomHexString(10);
    doc_version = 1;
  }

  int64_t total_len_inc = 0;

  for (size_t i = 0; i < chunks.size(); i++) {
    const ChunkId chunk_id = start_id + i;
    const int64_t chunk_len = chunk_lens.at(i);
    total_len_inc += chunk_len;

    ChunkData data{chunks.at(i), document, doc_id, doc_version};
    std::string chunk_data = serialize(data);

    std::string chunk_metadata = serialize(metadata.at(i));

    auto chunk_key = asSlice<ChunkId>(&chunk_id);

    std::array<rocksdb::Status, 3> statuses;
    statuses[0] =
        txn->Put(_chunk_counters, chunk_key, asSlice<int64_t>(&chunk_len));
    statuses[1] = txn->Put(_chunk_data, chunk_key, chunk_data);
    statuses[2] = txn->Put(_chunk_metadata, chunk_key, chunk_metadata);

    for (auto& status : statuses) {
      if (!status.ok()) {
        throw RocksdbError(status, "storing chunk metadata");
      }
    }
  }

  for (const auto& [token, chunk_counts] : token_counts) {
    const HashedToken token_value = token;
    const auto token_key = asSlice<HashedToken>(&token_value);
    const ChunkCountView view(chunk_counts);

    auto status = txn->Merge(_chunk_token_index, token_key, view.slice());
    if (!status.ok()) {
      throw RocksdbError(status, "inserting doc chunks");
    }
  }

  incrementCounter(txn, N_CHUNKS, chunks.size());

  incrementCounter(txn, TOTAL_CHUNK_LEN, total_len_inc);

  DocChunkRange doc_chunks{start_id, start_id + chunks.size()};
  auto status = txn->Put(_doc_chunks, docVersionKey(doc_id, doc_version),
                         asSlice<DocChunkRange>(&doc_chunks));
  if (!status.ok()) {
    throw RocksdbError(status, "storing doc metadata");
  }

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(status, "committing insertion");
  }
}

ChunkId OnDiskNeuralDB::reserveChunkIds(ChunkId n_ids) {
  auto txn = newTxn();

  std::string value;

  int64_t next_id;
  auto get_status = txn->GetForUpdate(rocksdb::ReadOptions(), _chunk_counters,
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
      txn->Put(_chunk_counters, NEXT_CHUNK_ID, asSlice<int64_t>(&new_next_id));
  if (!put_status.ok()) {
    throw RocksdbError(put_status, "reserving chunk ids for doc");
  }

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "committing chunk id reservation");
  }

  return next_id;
}

uint32_t OnDiskNeuralDB::getDocVersion(TxnPtr& txn, const std::string& doc_id) {
  std::string value;

  uint32_t version;
  auto get_status =
      txn->GetForUpdate(rocksdb::ReadOptions(), _doc_version, doc_id, &value);

  if (get_status.ok()) {
    if (value.size() != sizeof(uint32_t)) {
      throw NeuralDbError(ErrorCode::MalformedData, "doc version malformed");
    }
    version = *reinterpret_cast<uint32_t*>(value.data());
  }
  if (get_status.IsNotFound()) {
    version = 0;
  } else {
    throw RocksdbError(get_status, "retrieving doc version");
  }

  version++;

  auto put_status = txn->Put(_doc_version, doc_id, asSlice<uint32_t>(&version));
  if (!put_status.ok()) {
    throw RocksdbError(put_status, "updating doc version");
  }

  return version;
}

void OnDiskNeuralDB::incrementCounter(TxnPtr& txn, const std::string& key,
                                      int64_t value) {
  auto status = txn->Merge(_chunk_counters, key, asSlice<int64_t>(&value));
  if (!status.ok()) {
    throw RocksdbError(status, "incrementing counter");
  }
}

int64_t OnDiskNeuralDB::getCounter(const rocksdb::Slice& key) {
  std::string value;

  auto status = _db->Get(rocksdb::ReadOptions(), _chunk_counters, key, &value);
  if (!status.ok()) {
    throw RocksdbError(status, "retrieving counter");
  }

  if (value.size() != sizeof(int64_t)) {
    throw NeuralDbError(ErrorCode::MalformedData, "counter value malformed");
  }

  int64_t counter = *reinterpret_cast<int64_t*>(value.data());
  return counter;
}

template <typename T>
std::vector<std::optional<T>> OnDiskNeuralDB::loadChunkField(
    rocksdb::ColumnFamilyHandle* column,
    const std::vector<ChunkId>& chunk_ids) {
  std::vector<rocksdb::Slice> keys;
  keys.reserve(chunk_ids.size());

  for (const auto& id : chunk_ids) {
    keys.emplace_back(asSlice<ChunkId>(&id));
  }

  std::vector<rocksdb::PinnableSlice> values(keys.size());
  std::vector<rocksdb::Status> statuses(keys.size());

  _db->MultiGet(rocksdb::ReadOptions(), column, keys.size(), keys.data(),
                values.data(), statuses.data());

  std::vector<std::optional<T>> result;
  result.reserve(chunk_ids.size());

  for (size_t i = 0; i < chunk_ids.size(); i++) {
    if (statuses[i].ok()) {
      result.emplace_back(deserialize<T>(values[i]));
    } else if (statuses[i].IsNotFound()) {
      // This could happen if chunks are deleted between the first and second
      // phase of a query.
      result.emplace_back(std::nullopt);
    } else {
      throw RocksdbError(statuses[i], "retrieving chunk data");
    }
  }

  return result;
}

std::vector<rocksdb::PinnableSlice> OnDiskNeuralDB::mapTokensToChunks(
    const std::vector<HashedToken>& query_tokens) {
  std::vector<rocksdb::Slice> keys;
  keys.reserve(query_tokens.size());

  for (const HashedToken& token : query_tokens) {
    keys.emplace_back(asSlice<HashedToken>(&token));
  }
  std::vector<rocksdb::PinnableSlice> values(keys.size());
  std::vector<rocksdb::Status> statuses(keys.size());

  _db->MultiGet(rocksdb::ReadOptions(), _chunk_token_index, keys.size(),
                keys.data(), values.data(), statuses.data());

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

std::vector<std::pair<size_t, float>> OnDiskNeuralDB::rankByIdf(
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

std::unordered_map<ChunkId, float> OnDiskNeuralDB::candidateSet(
    const std::string& query) {
  const auto query_tokens = _text_processor.tokenize(query);

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

std::vector<std::pair<ChunkId, float>> topkCandidates(
    const std::unordered_map<ChunkId, float>& candidate_set, uint32_t top_k) {
  std::vector<std::pair<ChunkId, float>> heap;
  heap.reserve(top_k + 1);
  const HighestScore<ChunkId> cmp;

  for (const auto& [chunk, score] : candidate_set) {
    if (heap.size() < top_k || heap.front().second < score) {
      heap.emplace_back(chunk, score);
      std::push_heap(heap.begin(), heap.end(), cmp);
    }

    if (heap.size() > top_k) {
      std::pop_heap(heap.begin(), heap.end(), cmp);
      heap.pop_back();
    }
  }

  std::sort_heap(heap.begin(), heap.end(), cmp);

  return heap;
}

std::vector<std::pair<Chunk, float>> OnDiskNeuralDB::query(
    const std::string& query, uint32_t top_k) {
  auto candidate_set = candidateSet(query);

  const auto top_candidates = topkCandidates(candidate_set, top_k);

  std::vector<ChunkId> chunk_ids;
  chunk_ids.reserve(top_candidates.size());
  for (const auto& [chunk_id, _] : top_candidates) {
    chunk_ids.push_back(chunk_id);
  }

  const auto chunk_data = loadChunkField<ChunkData>(_chunk_data, chunk_ids);
  const auto metadata = loadChunkField<MetadataMap>(_chunk_metadata, chunk_ids);

  std::vector<std::pair<Chunk, float>> results;
  results.reserve(top_candidates.size());

  for (size_t i = 0; i < top_candidates.size(); i++) {
    if (!chunk_data[i] || !metadata[i]) {
      continue;  // metadata or fields are not found for chunk
    }
    const auto& data = chunk_data[i];
    results.emplace_back(
        Chunk(/*id=*/top_candidates[i].first, /*text=*/data->text,
              /*document=*/data->document, /*doc_id=*/data->doc_id,
              /*doc_version=*/data->doc_version, /*metadata=*/*metadata[i]),
        top_candidates[i].second);
  }

  return results;
}

std::vector<std::pair<ChunkId, float>> sortCandidates(
    const std::unordered_map<ChunkId, float>& candidate_set) {
  std::vector<std::pair<ChunkId, float>> candidates(candidate_set.begin(),
                                                    candidate_set.end());
  const HighestScore<ChunkId> cmp;
  std::sort(candidates.begin(), candidates.end(), cmp);

  return candidates;
}

std::vector<std::pair<Chunk, float>> OnDiskNeuralDB::rank(
    const std::string& query, const QueryConstraints& constraints,
    uint32_t top_k) {
  auto candidate_set = candidateSet(query);

  auto sorted_candidates = sortCandidates(candidate_set);

  std::vector<ChunkId> topk_chunk_ids;
  std::vector<ChunkId> topk_scores;
  std::vector<MetadataMap> topk_metadata;

  for (size_t start = 0; start < sorted_candidates.size(); start += top_k) {
    if (topk_chunk_ids.size() == top_k) {
      break;
    }

    size_t end = std::min(start + top_k, sorted_candidates.size());

    std::vector<ChunkId> chunk_ids(end - start);
    for (size_t i = start; i < end; i++) {
      chunk_ids[i - start] = sorted_candidates[i].first;
    }

    const auto metadata =
        loadChunkField<MetadataMap>(_chunk_metadata, chunk_ids);

    for (size_t i = 0; i < (end - start); i++) {
      std::cerr << "chunk: " << chunk_ids[i] << std::endl;
      if (metadata[i] && matches(constraints, *metadata[i])) {
        topk_chunk_ids.push_back(sorted_candidates[start + i].first);
        topk_scores.push_back(sorted_candidates[start + i].second);
        topk_metadata.push_back(metadata[i].value());
        if (topk_chunk_ids.size() == top_k) {
          break;
        }
      }
    }
  }

  const auto chunk_data =
      loadChunkField<ChunkData>(_chunk_data, topk_chunk_ids);

  std::vector<std::pair<Chunk, float>> results;
  results.reserve(topk_chunk_ids.size());

  for (size_t i = 0; i < topk_chunk_ids.size(); i++) {
    if (!chunk_data[i]) {
      continue;
    }
    results.emplace_back(
        Chunk(/*id=*/topk_chunk_ids[i], /*text=*/chunk_data[i]->text,
              /*document=*/chunk_data[i]->document,
              /*doc_id=*/chunk_data[i]->doc_id,
              /*doc_version=*/chunk_data[i]->doc_version,
              /*metadata=*/topk_metadata[i]),
        topk_scores[i]);
  }

  return results;
}

void OnDiskNeuralDB::deleteDoc(const DocId& doc, uint32_t version) {
  auto txn = newTxn();

  const auto doc_chunks = deleteDocChunkRange(txn, doc, version);

  std::vector<ChunkId> chunk_ids(doc_chunks.end - doc_chunks.start);
  std::iota(chunk_ids.begin(), chunk_ids.end(), doc_chunks.start);

  deleteChunkField(txn, _chunk_metadata, chunk_ids);
  deleteChunkField(txn, _chunk_data, chunk_ids);

  removeChunksFromIndex(txn, doc_chunks.start, doc_chunks.end);

  const int64_t deleted_len = deleteChunkLens(txn, chunk_ids);
  int64_t n_chunks = chunk_ids.size();
  incrementCounter(txn, N_CHUNKS, -n_chunks);
  incrementCounter(txn, TOTAL_CHUNK_LEN, -deleted_len);
}

OnDiskNeuralDB::DocChunkRange OnDiskNeuralDB::deleteDocChunkRange(
    TxnPtr& txn, const DocId& doc_id, uint32_t version) {
  std::string value;
  auto status = txn->GetForUpdate(rocksdb::ReadOptions(), _doc_chunks,
                                  docVersionKey(doc_id, version), &value);
  if (status.IsNotFound()) {
    throw NeuralDbError(ErrorCode::DocNotFound,
                        "document '" + doc_id + "' version " +
                            std::to_string(version) + " not found");
  }
  if (!status.ok()) {
    throw RocksdbError(status, "retrieving doc chunks");
  }

  DocChunkRange chunk_range = *reinterpret_cast<DocChunkRange*>(value.data());

  return chunk_range;
}

void OnDiskNeuralDB::removeChunksFromIndex(TxnPtr& txn, ChunkId start,
                                           ChunkId end) {
  auto iter = std::unique_ptr<rocksdb::Iterator>(
      txn->GetIterator(rocksdb::ReadOptions(), _chunk_token_index));

  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    ChunkCountView curr_value(iter->value());

    std::vector<ChunkCount> new_value;
    new_value.reserve(curr_value.size());

    for (const auto& count : curr_value) {
      if (count.chunk_id < start || end <= count.chunk_id) {
        new_value.push_back(count);
      }
    }

    if (new_value.size() < curr_value.size()) {
      ChunkCountView view(new_value);
      auto put_status = txn->Put(_chunk_token_index, iter->key(), view.slice());
      if (!put_status.ok()) {
        throw RocksdbError(put_status, "removing doc chunks");
      }
    }
  }
}

int64_t OnDiskNeuralDB::deleteChunkLens(TxnPtr& txn,
                                        const std::vector<ChunkId>& chunk_ids) {
  int64_t deleted_len = 0;

  for (const auto& chunk_id : chunk_ids) {
    const auto key = asSlice<ChunkId>(&chunk_id);
    std::string value;
    auto get_status =
        txn->GetForUpdate(rocksdb::ReadOptions(), _chunk_counters, key, &value);
    if (!get_status.ok()) {
      throw RocksdbError(get_status, "removing doc chunk data");
    }

    if (value.size() != sizeof(int64_t)) {
      throw NeuralDbError(ErrorCode::MalformedData, "counter is malformed");
    }

    deleted_len += *reinterpret_cast<int64_t*>(value.data());

    auto del_status = txn->Delete(_chunk_counters, key);
    if (!del_status.ok()) {
      throw RocksdbError(get_status, "removing doc chunk data");
    }
  }

  return deleted_len;
}

void OnDiskNeuralDB::deleteChunkField(TxnPtr& txn,
                                      rocksdb::ColumnFamilyHandle* column,
                                      const std::vector<ChunkId>& chunk_ids) {
  for (const auto& chunk_id : chunk_ids) {
    auto status = txn->Delete(column, asSlice<ChunkId>(&chunk_id));
    if (!status.ok()) {
      throw RocksdbError(status, "removing doc chunk metadata");
    }
  }
}

void OnDiskNeuralDB::prune() {
  const int64_t n_chunks = getCounter(N_CHUNKS);

  const size_t max_chunks_with_token =
      std::max<size_t>(_max_token_occurrence_frac * n_chunks, 1000);

  auto txn = newTxn();

  auto iter = std::unique_ptr<rocksdb::Iterator>(
      txn->GetIterator(rocksdb::ReadOptions(), _chunk_token_index));

  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    ChunkCountView chunks(iter->value());
    if (chunks.size() > max_chunks_with_token) {
      auto status = txn->Put(_chunk_token_index, iter->key(),
                             asSlice<ChunkCount>(&PRUNED));
      if (!status.ok()) {
        throw RocksdbError(status, "pruning db");
      }
    }
  }

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "committing prune");
  }

  auto compact = _db->CompactRange(rocksdb::CompactRangeOptions(),
                                   _chunk_token_index, nullptr, nullptr);
  if (!compact.ok()) {
    throw RocksdbError(compact, "compacting db");
  }
}

TxnPtr OnDiskNeuralDB::newTxn() {
  if (!_transaction_db) {
    throw NeuralDbError(
        ErrorCode::ReadOnly,
        "cannot perform this operation since neural db is in read only mode");
  }
  return TxnPtr(_transaction_db->BeginTransaction(rocksdb::WriteOptions()));
}

OnDiskNeuralDB::~OnDiskNeuralDB() {
  _db->DestroyColumnFamilyHandle(_default);
  _db->DestroyColumnFamilyHandle(_chunk_counters);
  _db->DestroyColumnFamilyHandle(_chunk_data);
  _db->DestroyColumnFamilyHandle(_chunk_metadata);
  _db->DestroyColumnFamilyHandle(_chunk_token_index);
  _db->DestroyColumnFamilyHandle(_doc_chunks);
  _db->DestroyColumnFamilyHandle(_doc_version);
  _db->Close();
  delete _db;
}

}  // namespace thirdai::search::ndb
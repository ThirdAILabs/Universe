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
#include <search/src/neural_db/on_disk/RocksDBError.h>
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

std::string docVersionKey(const std::string& doc_id, uint32_t doc_version) {
  return doc_id + "_" + std::to_string(doc_version);
}

OnDiskNeuralDB::OnDiskNeuralDB(const std::string& save_path,
                               const IndexConfig& config, bool read_only)
    : _save_path(save_path), _text_processor(config.tokenizer) {
  licensing::checkLicense();

  createDirectory(save_path);

  rocksdb::Options options;
  options.create_if_missing = true;
  options.create_missing_column_families = true;

  rocksdb::ColumnFamilyOptions counter_options;
  counter_options.merge_operator = std::make_shared<IncrementCounter>();

  rocksdb::ColumnFamilyOptions concat_counts_options;
  concat_counts_options.merge_operator = std::make_shared<ConcatChunkCounts>();

  rocksdb::ColumnFamilyOptions concat_options;
  concat_options.merge_operator = std::make_shared<Concat>();

  std::vector<rocksdb::ColumnFamilyDescriptor> column_families = {
      {rocksdb::kDefaultColumnFamilyName, {}},
      {"chunk_counters", counter_options},
      {"chunk_index", concat_counts_options},
      {"chunk_data", {}},
      {"chunk_metadata", {}},
      {"doc_chunks", {}},
      {"doc_version", {}},
      {"query_counters", counter_options},
      {"query_index", concat_counts_options},
      {"id_map", concat_options}};

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
  _chunk_index = std::make_unique<InvertedIndex>(
      _db, columns.at(1), columns.at(2), config, read_only);
  _chunk_data = columns.at(3);
  _chunk_metadata = columns.at(4);
  _doc_chunks = columns.at(5);
  _doc_version = columns.at(6);
  _query_index = std::make_unique<InvertedIndex>(
      _db, columns.at(7), columns.at(8), config, read_only);
  _query_to_chunks = columns.at(9);
}

void OnDiskNeuralDB::insert(const std::string& document,
                            const std::vector<std::string>& chunks,
                            const std::vector<MetadataMap>& metadata,
                            const std::optional<std::string>& doc_id_opt) {
  if (chunks.size() != metadata.size()) {
    throw std::invalid_argument("length of metadata and chunks must match");
  }

  auto initTxn = newTxn();  // this will be committed when by the index
  const ChunkId start_id =
      _chunk_index->reserveChunkIds(initTxn, chunks.size());

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

  for (size_t i = 0; i < chunks.size(); i++) {
    const ChunkId chunk_id = start_id + i;
    const auto chunk_key = asSlice<ChunkId>(&chunk_id);

    ChunkData data{chunks.at(i), document, doc_id, doc_version};
    std::string chunk_data = serialize(data);

    std::string chunk_metadata = serialize(metadata.at(i));

    std::array<rocksdb::Status, 2> statuses;
    statuses[0] = txn->Put(_chunk_data, chunk_key, chunk_data);
    statuses[1] = txn->Put(_chunk_metadata, chunk_key, chunk_metadata);

    for (auto& status : statuses) {
      if (!status.ok()) {
        throw RocksdbError(status, "storing chunk metadata");
      }
    }
  }

  _chunk_index->insert(txn, start_id, token_counts, chunk_lens);

  DocChunkRange doc_chunks{start_id, start_id + chunks.size()};
  auto status = txn->Put(_doc_chunks, docVersionKey(doc_id, doc_version),
                         asSlice<DocChunkRange>(&doc_chunks));
  if (!status.ok()) {
    throw RocksdbError(status, "storing doc metadata");
  }

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "committing insertion");
  }
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
  auto candidate_set =
      _chunk_index->candidateSet(_text_processor.tokenize(query));

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
  auto candidate_set =
      _chunk_index->candidateSet(_text_processor.tokenize(query));

  auto sorted_candidates = sortCandidates(candidate_set);

  std::vector<ChunkId> topk_chunk_ids;
  std::vector<ChunkId> topk_scores;
  std::vector<MetadataMap> topk_metadata;

  const size_t batch_size = 20;
  for (size_t start = 0; start < sorted_candidates.size();
       start += batch_size) {
    if (topk_chunk_ids.size() == top_k) {
      break;
    }

    size_t end = std::min(start + batch_size, sorted_candidates.size());

    std::vector<ChunkId> chunk_ids(end - start);
    for (size_t i = start; i < end; i++) {
      chunk_ids[i - start] = sorted_candidates[i].first;
    }

    const auto metadata =
        loadChunkField<MetadataMap>(_chunk_metadata, chunk_ids);

    for (size_t i = 0; i < (end - start); i++) {
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

void OnDiskNeuralDB::finetune(
    const std::vector<std::vector<ChunkId>>& chunk_ids,
    const std::vector<std::string>& queries) {
  if (chunk_ids.size() != queries.size()) {
    throw std::invalid_argument(
        "number of labels must match number of queries for finetuning.");
  }

  auto initTxn = newTxn();  // this will be committed when by the index
  const ChunkId start_id =
      _query_index->reserveChunkIds(initTxn, queries.size());

  auto [token_counts, chunk_lens] = _text_processor.process(start_id, queries);

  auto txn = newTxn();

  _query_index->insert(txn, start_id, token_counts, chunk_lens);

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "committing insertion");
  }
}

void OnDiskNeuralDB::deleteDoc(const DocId& doc, uint32_t version) {
  auto txn = newTxn();

  const auto doc_chunks = deleteDocChunkRange(txn, doc, version);

  std::vector<ChunkId> chunk_ids(doc_chunks.end - doc_chunks.start);
  std::iota(chunk_ids.begin(), chunk_ids.end(), doc_chunks.start);

  deleteChunkField(txn, _chunk_metadata, chunk_ids);
  deleteChunkField(txn, _chunk_data, chunk_ids);

  _chunk_index->deleteChunks(txn, {chunk_ids.begin(), chunk_ids.end()});

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "deleting doc");
  }
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
  auto txn = newTxn();
  _chunk_index->prune(txn);  // txn is commit by index
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
  _chunk_index.reset();  // force destructor to run before db is closed here.
  _query_index.reset();  // force destructor to run before db is closed here.

  _db->DestroyColumnFamilyHandle(_default);
  _db->DestroyColumnFamilyHandle(_chunk_data);
  _db->DestroyColumnFamilyHandle(_chunk_metadata);
  _db->DestroyColumnFamilyHandle(_doc_chunks);
  _db->DestroyColumnFamilyHandle(_doc_version);

  _db->Close();
  delete _db;
}

}  // namespace thirdai::search::ndb
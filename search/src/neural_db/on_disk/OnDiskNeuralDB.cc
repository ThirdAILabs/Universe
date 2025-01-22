#include "OnDiskNeuralDB.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <licensing/src/CheckLicense.h>
#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/status.h>
#include <search/src/inverted_index/BM25.h>
#include <search/src/inverted_index/IndexConfig.h>
#include <search/src/inverted_index/Utils.h>
#include <search/src/neural_db/Chunk.h>
#include <search/src/neural_db/Constraints.h>
#include <search/src/neural_db/Errors.h>
#include <search/src/neural_db/on_disk/DataView.h>
#include <search/src/neural_db/on_disk/MergeOperators.h>
#include <search/src/neural_db/on_disk/RocksDBError.h>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::search::ndb {

// Finetuning parameters
constexpr size_t QUERY_INDEX_THRESHOLD = 10;
constexpr size_t TOP_QUERIES = 10;
constexpr float LAMBDA = 0.6;

constexpr char DOC_VER_DELIMITER = ';';

static std::string dbPath(const std::string& base) {
  return (std::filesystem::path(base) / "model").string();
}

static std::string metadataPath(const std::string& base) {
  return (std::filesystem::path(base) / "metadata").string();
}

std::string docVersionKey(const std::string& doc_id, uint32_t doc_version) {
  return doc_id + DOC_VER_DELIMITER + std::to_string(doc_version);
}

rocksdb::Options dbOptions() {
  rocksdb::Options options;
  options.create_if_missing = true;
  options.create_missing_column_families = true;

  return options;
}

std::vector<rocksdb::ColumnFamilyDescriptor> columnFamilies() {
  rocksdb::ColumnFamilyOptions counter_options;
  counter_options.merge_operator = std::make_shared<IncrementCounter>();

  rocksdb::ColumnFamilyOptions concat_counts_options;
  concat_counts_options.merge_operator = std::make_shared<ConcatChunkCounts>();

  rocksdb::ColumnFamilyOptions concat_options;
  concat_options.merge_operator = std::make_shared<Concat>();

  return {
      {rocksdb::kDefaultColumnFamilyName, {}},
      {"chunk_counters", counter_options},
      {"chunk_index", concat_counts_options},
      {"chunk_data", {}},
      {"chunk_metadata", {}},
      {"doc_chunks", concat_options},
      {"doc_version", {}},
      {"doc_id_to_name", {}},
      {"query_counters", counter_options},
      {"query_index", concat_counts_options},
      {"id_map", concat_options},
  };
}

std::unique_ptr<OnDiskNeuralDB> OnDiskNeuralDB::make(
    const std::string& save_path) {
  return std::make_unique<OnDiskNeuralDB>(save_path);
}

OnDiskNeuralDB::OnDiskNeuralDB(const std::string& save_path)
    : OnDiskNeuralDB(save_path, IndexConfig(), false) {}

OnDiskNeuralDB::OnDiskNeuralDB(const std::string& save_path,
                               const IndexConfig& config, bool read_only)
    : _save_path(save_path), _text_processor(config.tokenizer) {
  licensing::checkLicense();

  std::string db_path = dbPath(save_path);
  createDirectory(db_path);

  if (!std::filesystem::exists(metadataPath(save_path))) {
    auto metadata = ar::Map::make();
    metadata->set("config", config.toArchive());

    auto metadata_file = dataset::SafeFileIO::ofstream(metadataPath(save_path));
    ar::serialize(metadata, metadata_file);
  }

  std::vector<rocksdb::ColumnFamilyHandle*> columns;

  rocksdb::Status open_status;
  if (!read_only) {
    open_status = rocksdb::TransactionDB::Open(
        dbOptions(), rocksdb::TransactionDBOptions(), db_path, columnFamilies(),
        &columns, &_transaction_db);
    _db = _transaction_db;
  } else {
    open_status = rocksdb::DB::OpenForReadOnly(
        dbOptions(), db_path, columnFamilies(), &columns, &_db);
    _transaction_db = nullptr;
  }
  if (!open_status.ok()) {
    throw RocksdbError(open_status, "unable to open database");
  }

  _default = columns.at(0);
  _chunk_index = std::make_unique<InvertedIndex>(
      _db, columns.at(1), columns.at(2), config, read_only);

  _chunk_data =
      std::make_unique<ChunkDataColumn<ChunkData>>(_db, columns.at(3));
  _chunk_metadata =
      std::make_unique<ChunkDataColumn<MetadataMap>>(_db, columns.at(4));
  _doc_chunks = columns.at(5);
  _doc_version = columns.at(6);
  _doc_id_to_name = columns.at(7);
  _query_index = std::make_unique<InvertedIndex>(
      _db, columns.at(8), columns.at(9), config, read_only);
  _query_to_chunks = std::make_unique<QueryToChunks>(_db, columns.at(10));
}

InsertMetadata OnDiskNeuralDB::insert(const std::vector<std::string>& chunks,
                                      const std::vector<MetadataMap>& metadata,
                                      const std::string& document,
                                      const DocId& doc_id,
                                      std::optional<uint32_t> doc_version_opt) {
  if (chunks.size() != metadata.size()) {
    throw std::invalid_argument("length of metadata and chunks must match");
  }

  if (doc_id.find_first_of(DOC_VER_DELIMITER) != std::string::npos) {
    throw std::invalid_argument("doc id cannot contain ';'");
  }

  auto initTxn = newTxn();  // this will be committed by the index
  const ChunkId start_id =
      _chunk_index->reserveChunkIds(initTxn, chunks.size());

  auto [token_counts, chunk_lens] = _text_processor.process(start_id, chunks);

  /**
   * We use 2 transactions becuase we need the reserved chunk ids to do the
   * initial text processing/frequency counting, but it its not ideal to have a
   * transaction open while doing that processing. This is still safe because if
   * the second transaction fails the only risk is those chunk_ids are lost,
   * which is not a big concern. Additionally, reserving chunk ids are the main
   * place where we could have transaction conflicts (except for doc versions)
   * so it is unlikely for the second transaction to fail with a conflict.
   */
  auto txn = newTxn();

  uint32_t doc_version =
      doc_version_opt ? *doc_version_opt : getDocVersion(txn, doc_id);

  std::vector<ChunkData> chunk_data;
  chunk_data.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    chunk_data.emplace_back(chunk, document, doc_id, doc_version);
  }

  _chunk_data->write(txn, start_id, chunk_data);

  _chunk_metadata->write(txn, start_id, metadata);

  _chunk_index->insert(txn, start_id, token_counts, chunk_lens);

  const auto doc_ver_key = docVersionKey(doc_id, doc_version);

  DocChunkRange doc_chunks{start_id, start_id + chunks.size()};
  auto chunk_range_status =
      txn->Merge(_doc_chunks, doc_ver_key, asSlice<DocChunkRange>(&doc_chunks));
  if (!chunk_range_status.ok()) {
    throw RocksdbError(chunk_range_status, "updating doc chunk mapping");
  }

  auto doc_name_status = txn->Put(_doc_id_to_name, doc_ver_key, document);
  if (!doc_name_status.ok()) {
    throw RocksdbError(doc_name_status, "updating doc name mapping");
  }

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "committing insertion");
  }

  return InsertMetadata(doc_id, doc_version, start_id,
                        start_id + chunks.size());
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
  } else if (get_status.IsNotFound()) {
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

std::unordered_map<ChunkId, float> OnDiskNeuralDB::candidateSet(
    const std::string& query) {
  const auto query_tokens = _text_processor.tokenize(query);

  auto candidate_set = _chunk_index->candidateSet(query_tokens);

  if (_query_index->size() == 0) {
    return candidate_set;
  }

  for (auto& [_, score] : candidate_set) {
    score *= LAMBDA;
  }

  auto top_queries = topkCandidates(
      _query_index->candidateSet(query_tokens, QUERY_INDEX_THRESHOLD),
      TOP_QUERIES);

  for (const auto& [query_id, score] : top_queries) {
    for (const ChunkId chunk_id : _query_to_chunks->getChunks(query_id)) {
      candidate_set[chunk_id] += (1 - LAMBDA) * score;
    }
  }

  return candidate_set;
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

  const auto chunk_data = _chunk_data->get(chunk_ids);
  const auto metadata = _chunk_metadata->get(chunk_ids);

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

  const size_t batch_size = 20;
  for (size_t start = 0;
       start < sorted_candidates.size() && topk_chunk_ids.size() < top_k;
       start += batch_size) {
    size_t end = std::min(start + batch_size, sorted_candidates.size());

    std::vector<ChunkId> chunk_ids(end - start);
    for (size_t i = start; i < end; i++) {
      chunk_ids[i - start] = sorted_candidates[i].first;
    }

    const auto metadata = _chunk_metadata->get(chunk_ids);

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

  const auto chunk_data = _chunk_data->get(topk_chunk_ids);

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
    const std::vector<std::string>& queries,
    const std::vector<std::vector<ChunkId>>& chunk_ids) {
  if (chunk_ids.size() != queries.size()) {
    throw std::invalid_argument(
        "number of labels must match number of queries for finetuning.");
  }

  // See comment above for why we have 2 transactions.
  auto initTxn = newTxn();  // this will be committed by the index
  const ChunkId start_id =
      _query_index->reserveChunkIds(initTxn, queries.size());

  auto [token_counts, chunk_lens] = _text_processor.process(start_id, queries);

  auto txn = newTxn();

  _query_index->insert(txn, start_id, token_counts, chunk_lens);

  _query_to_chunks->addQueries(txn, start_id, chunk_ids);

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "committing insertion");
  }
}

void OnDiskNeuralDB::associate(const std::vector<std::string>& sources,
                               const std::vector<std::string>& targets,
                               uint32_t strength) {
  if (sources.size() != targets.size()) {
    throw std::invalid_argument(
        "number of sources must match number of targets for associate.");
  }

  std::vector<std::vector<ChunkId>> labels;
  labels.reserve(sources.size());

  for (const auto& target : targets) {
    auto results = query(target, strength);
    std::vector<ChunkId> ids;
    ids.reserve(results.size());
    for (const auto& res : results) {
      ids.push_back(res.first.id);
    }
    labels.push_back(std::move(ids));
  }

  finetune(sources, labels);
}

void OnDiskNeuralDB::deleteDocVersion(const DocId& doc_id,
                                      uint32_t doc_version) {
  auto txn = newTxn();

  const auto doc_chunks = deleteDocChunkRangesAndName(txn, doc_id, doc_version);

  _chunk_data->remove(txn, doc_chunks);
  _chunk_metadata->remove(txn, doc_chunks);

  _chunk_index->deleteChunks(txn, doc_chunks);
  _query_to_chunks->deleteChunks(txn, doc_chunks);

  auto commit = txn->Commit();
  if (!commit.ok()) {
    throw RocksdbError(commit, "deleting doc");
  }
}

std::unordered_set<ChunkId> OnDiskNeuralDB::deleteDocChunkRangesAndName(
    TxnPtr& txn, const DocId& doc_id, uint32_t version) {
  const auto doc_ver_key = docVersionKey(doc_id, version);
  std::string value;
  auto status = txn->GetForUpdate(rocksdb::ReadOptions(), _doc_chunks,
                                  doc_ver_key, &value);
  if (status.IsNotFound()) {
    throw NeuralDbError(ErrorCode::DocNotFound,
                        "document '" + doc_id + "' version " +
                            std::to_string(version) + " not found");
  }
  if (!status.ok()) {
    throw RocksdbError(status, "retrieving doc chunks");
  }

  DataView<DocChunkRange> ranges(value);

  std::unordered_set<ChunkId> deleted_ids;
  for (const auto& range : ranges) {
    for (ChunkId i = range.start; i < range.end; i++) {
      deleted_ids.insert(i);
    }
  }

  auto del_ranges = txn->Delete(_doc_chunks, doc_ver_key);
  if (!del_ranges.ok()) {
    throw RocksdbError(del_ranges, "deleting entry from doc chunks mapping");
  }

  auto del_name = txn->Delete(_doc_id_to_name, doc_ver_key);
  if (!del_name.ok()) {
    throw RocksdbError(del_name, "deleting entry from doc name mapping");
  }

  return deleted_ids;
}

void OnDiskNeuralDB::deleteDoc(const DocId& doc_id, bool keep_latest_version) {
  auto versions = getDocVersions(doc_id);

  if (versions.empty()) {
    return;
  }

  uint32_t maxVersion = versions[0];
  for (uint32_t version : versions) {
    if (version > maxVersion) {
      maxVersion = version;
    }
  }

  for (uint32_t version : versions) {
    if (keep_latest_version && version == maxVersion) {
      continue;
    }

    try {
      deleteDocVersion(doc_id, version);
    } catch (const NeuralDbError& e) {
      if (e.code() != ErrorCode::DocNotFound) {
        // DocNotFound is ok, possible with concurrent deletets. Rethrow other
        // errors.
        throw;
      }
    }
  }
}

std::pair<DocId, uint32_t> parseDocIdAndVersion(const std::string& key) {
  auto loc = key.find_first_of(DOC_VER_DELIMITER);
  if (loc == std::string::npos) {
    throw NeuralDbError(ErrorCode::MalformedData,
                        "invalid document version key");
  }

  const std::string doc_id = key.substr(0, loc);
  uint32_t version;
  try {
    version = std::stoul(key.substr(loc + 1));
  } catch (const std::invalid_argument& e) {  // parse error
    throw NeuralDbError(ErrorCode::MalformedData,
                        "invalid document version key");
  }

  return {doc_id, version};
}

std::vector<uint32_t> OnDiskNeuralDB::getDocVersions(const DocId& doc_id) {
  auto iter = std::unique_ptr<rocksdb::Iterator>(
      _db->NewIterator(rocksdb::ReadOptions(), _doc_id_to_name));

  std::vector<uint32_t> versions;
  for (iter->Seek(doc_id); iter->Valid() && iter->key().starts_with(doc_id);
       iter->Next()) {
    auto key = iter->key().ToString();

    const auto [parsed_doc_id, version] = parseDocIdAndVersion(key);

    if (parsed_doc_id == doc_id) {
      versions.push_back(version);
    }
  }

  return versions;
}

void OnDiskNeuralDB::prune() {
  auto txn = newTxn();
  _chunk_index->prune(txn);  // txn is commit by index
}

std::vector<Source> OnDiskNeuralDB::sources() {
  auto iter = std::unique_ptr<rocksdb::Iterator>(
      _db->NewIterator(rocksdb::ReadOptions(), _doc_id_to_name));

  std::vector<Source> sources;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    auto key = iter->key().ToString();
    const auto [doc_id, version] = parseDocIdAndVersion(key);
    sources.emplace_back(iter->value().ToString(), doc_id, version);
  }

  return sources;
}

void OnDiskNeuralDB::save(const std::string& save_path) const {
  licensing::entitlements().verifySaveLoad();

  createDirectory(save_path);

  rocksdb::Checkpoint* ckpt_ptr;
  auto ckpt_create_status = rocksdb::Checkpoint::Create(_db, &ckpt_ptr);

  // Convert to unique pointer for better memory management
  auto ckpt = std::unique_ptr<rocksdb::Checkpoint>(ckpt_ptr);

  if (!ckpt_create_status.ok()) {
    throw RocksdbError(ckpt_create_status, "creating checkpoint");
  }

  auto ckpt_save_status = ckpt->CreateCheckpoint(dbPath(save_path));
  if (!ckpt_save_status.ok()) {
    throw RocksdbError(ckpt_save_status, "saving checkpoint");
  }

  std::filesystem::copy(metadataPath(_save_path), metadataPath(save_path));
}

std::shared_ptr<OnDiskNeuralDB> OnDiskNeuralDB::load(
    const std::string& save_path, bool read_only) {
  licensing::entitlements().verifySaveLoad();

  auto metadata_file = dataset::SafeFileIO::ifstream(metadataPath(save_path));
  auto metadata = ar::deserialize(metadata_file);

  auto config = IndexConfig::fromArchive(*metadata->get("config"));

  return std::make_shared<OnDiskNeuralDB>(save_path, config, read_only);
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
  // These are to force the destructors to run before db is closed here.
  _chunk_index.reset();
  _query_index.reset();
  _chunk_data.reset();
  _chunk_metadata.reset();
  _query_to_chunks.reset();

  _db->DestroyColumnFamilyHandle(_default);
  _db->DestroyColumnFamilyHandle(_doc_chunks);
  _db->DestroyColumnFamilyHandle(_doc_version);
  _db->DestroyColumnFamilyHandle(_doc_id_to_name);

  _db->Close();
  delete _db;
}

}  // namespace thirdai::search::ndb
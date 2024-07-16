#include "ShardedRetriever.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <search/src/inverted_index/IndexConfig.h>
#include <search/src/inverted_index/OnDiskIndex.h>
#include <search/src/inverted_index/Retriever.h>
#include <search/src/inverted_index/Utils.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::search {

class OnDiskFactory final : public RetrieverFactory {
 public:
  explicit OnDiskFactory(std::string save_path, bool read_only)
      : _save_path(std::move(save_path)), _read_only(read_only) {}

  std::shared_ptr<Retriever> create(const IndexConfig& config,
                                    size_t shard_id) const final {
    if (_read_only) {
      throw std::invalid_argument(
          "Cannot create new shards in read only mode.");
    }
    return std::make_shared<OnDiskIndex>(
        std::filesystem::path(_save_path) /
            ("shard_" + std::to_string(shard_id)),
        config);
  }

  std::shared_ptr<Retriever> load(const std::string& path) const final {
    return OnDiskIndex::load(path, _read_only);
  }

  std::string type() const final { return OnDiskIndex::typeName(); }

 private:
  std::string _save_path;
  bool _read_only;
};

class InMemoryFactory final : public RetrieverFactory {
 public:
  std::shared_ptr<Retriever> create(const IndexConfig& config,
                                    size_t shard_id) const final {
    (void)shard_id;
    return std::make_shared<InvertedIndex>(config);
  }

  std::shared_ptr<Retriever> load(const std::string& path) const final {
    return InvertedIndex::load(path);
  }

  std::string type() const final { return InvertedIndex::typeName(); }
};

ShardedRetriever::ShardedRetriever(IndexConfig config,
                                   const std::optional<std::string>& save_path)
    : _config(std::move(config)), _shard_size(config.shard_size) {
  if (save_path) {
    _factory = std::make_shared<OnDiskFactory>(*save_path, /*read_only=*/false);
  } else {
    _factory = std::make_shared<InMemoryFactory>();
  }
  _shards.push_back(_factory->create(_config, /*shard_id=*/0));
}

void ShardedRetriever::index(const std::vector<DocId>& ids,
                             const std::vector<std::string>& docs) {
  size_t indexed = 0;

  while (indexed < ids.size()) {
    if (_shards.back()->size() >= _shard_size) {
      _shards.push_back(_factory->create(_config, _shards.size()));
    }
    auto& curr_shard = _shards.back();
    size_t batchsize =
        std::min(_shard_size - curr_shard->size(), ids.size() - indexed);

    curr_shard->index(
        {ids.begin() + indexed, ids.begin() + indexed + batchsize},
        {docs.begin() + indexed, docs.begin() + indexed + batchsize});

    indexed += batchsize;
  }
}

std::vector<DocScore> topkAcrossShards(
    const std::vector<std::vector<DocScore>>& results, uint32_t k) {
  std::vector<DocScore> top_scores;
  top_scores.reserve(k + 1);
  const HighestScore<DocId> cmp;

  for (const auto& shard_results : results) {
    for (const auto& [doc, score] : shard_results) {
      if (top_scores.size() < k || top_scores.front().second < score) {
        top_scores.emplace_back(doc, score);
        std::push_heap(top_scores.begin(), top_scores.end(), cmp);
      }

      if (top_scores.size() > k) {
        std::pop_heap(top_scores.begin(), top_scores.end(), cmp);
        top_scores.pop_back();
      }
    }
  }

  std::sort_heap(top_scores.begin(), top_scores.end(), cmp);

  return top_scores;
}

std::vector<DocScore> ShardedRetriever::query(const std::string& query,
                                              uint32_t k,
                                              bool parallelize) const {
  if (_shards.size() == 1) {
    return _shards[0]->query(query, k, parallelize);
  }

  std::vector<std::vector<DocScore>> results(_shards.size());

#pragma omp parallel for default(none) \
    shared(results, query, k) if (parallelize)
  for (size_t i = 0; i < _shards.size(); i++) {
    results[i] = _shards[i]->query(query, k, /*parallelize=*/false);
  }

  return topkAcrossShards(results, k);
}

std::vector<DocScore> ShardedRetriever::rank(
    const std::string& query, const std::unordered_set<DocId>& candidates,
    uint32_t k, bool parallelize) const {
  if (_shards.size() == 1) {
    return _shards[0]->rank(query, candidates, k, parallelize);
  }

  std::vector<std::vector<DocScore>> results(_shards.size());

#pragma omp parallel for default(none) \
    shared(results, query, candidates, k) if (parallelize)
  for (size_t i = 0; i < _shards.size(); i++) {
    results[i] = _shards[i]->rank(query, candidates, k, /*parallelize=*/false);
  }

  return topkAcrossShards(results, k);
}

void ShardedRetriever::remove(const std::vector<DocId>& ids) {
  for (const auto& shard : _shards) {
    shard->remove(ids);
  }
}

size_t ShardedRetriever::size() const {
  size_t total_size = 0;
  for (const auto& shard : _shards) {
    total_size += shard->size();
  }
  return total_size;
}

void ShardedRetriever::prune() {
  for (const auto& shard : _shards) {
    shard->prune();
  }
}

void ShardedRetriever::save(const std::string& new_save_path) const {
  std::vector<std::string> saved_shards;
  for (size_t i = 0; i < _shards.size(); i++) {
    std::string shard_file = "shard_" + std::to_string(i);
    _shards[i]->save(
        (std::filesystem::path(new_save_path) / shard_file).string());
    saved_shards.push_back(shard_file);
  }

  auto metadata = ar::Map::make();
  metadata->set("type", ar::str(_factory->type()));
  metadata->set("config", _config.toArchive());
  metadata->set("shards", ar::vecStr(saved_shards));
  metadata->set("shard_size", ar::u64(_shard_size));

  auto metadata_file = dataset::SafeFileIO::ofstream(
      (std::filesystem::path(new_save_path) / "metadata").string());
  ar::serialize(metadata, metadata_file);
}

ShardedRetriever::ShardedRetriever(const std::string& save_path,
                                   bool read_only) {
  auto metadata_file = dataset::SafeFileIO::ifstream(
      (std::filesystem::path(save_path) / "metadata").string());
  auto metadata = ar::deserialize(metadata_file);

  _shard_size = metadata->u64("shard_size");

  _config = IndexConfig::fromArchive(*metadata->get("config"));

  auto type = metadata->str("type");

  if (type == OnDiskIndex::typeName()) {
    _factory = std::make_shared<OnDiskFactory>(save_path, read_only);
  } else if (type == InvertedIndex::typeName()) {
    _factory = std::make_shared<InMemoryFactory>();
  } else {
    throw std::invalid_argument("Invalid factory type: '" + type + "'.");
  }

  for (const auto& shard_file : metadata->getAs<ar::VecStr>("shards")) {
    _shards.push_back(_factory->load(
        (std::filesystem::path(save_path) / shard_file).string()));
  }
}

std::shared_ptr<ShardedRetriever> ShardedRetriever::load(
    const std::string& save_path, bool read_only) {
  return std::shared_ptr<ShardedRetriever>(
      new ShardedRetriever(save_path, read_only));
}

}  // namespace thirdai::search
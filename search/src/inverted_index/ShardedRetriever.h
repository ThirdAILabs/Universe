#pragma once

#include <archive/src/Archive.h>
#include <search/src/inverted_index/IndexConfig.h>
#include <search/src/inverted_index/Retriever.h>
#include <vector>

namespace thirdai::search {

class RetrieverFactory {
 public:
  virtual std::shared_ptr<Retriever> create(const IndexConfig& config,
                                            size_t shard_id) const = 0;

  virtual std::shared_ptr<Retriever> load(const std::string& path) const = 0;

  virtual std::string type() const = 0;

  virtual ~RetrieverFactory() = default;
};

class ShardedRetriever final : public Retriever {
 public:
  ShardedRetriever(IndexConfig config,
                   const std::optional<std::string>& save_path);

  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs) final;

  void update(const std::vector<DocId>& ids,
              const std::vector<std::string>& extra_tokens) final;

  std::vector<DocScore> query(const std::string& query, uint32_t k,
                              bool parallelize) const final;

  std::vector<DocScore> rank(const std::string& query,
                             const std::unordered_set<DocId>& candidates,
                             uint32_t k, bool parallelize) const final;

  void remove(const std::vector<DocId>& ids) final;

  size_t size() const final;

  void prune() final;

  void save(const std::string& new_save_path) const final;

  static std::shared_ptr<ShardedRetriever> load(const std::string& save_path,
                                                bool read_only);

  std::string type() const final { return typeName(); }

  static std::string typeName() { return "sharded-retriever"; }

 private:
  explicit ShardedRetriever(const std::string& save_path, bool read_only);

  std::shared_ptr<RetrieverFactory> _factory;
  IndexConfig _config;

  std::vector<std::shared_ptr<Retriever>> _shards;
  size_t _shard_size;
};

}  // namespace thirdai::search
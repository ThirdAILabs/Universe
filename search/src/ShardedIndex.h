#pragma once

#include <archive/src/Archive.h>
#include <search/src/InvertedIndex.h>

namespace thirdai::search {

class ShardedIndex {
 public:
  static constexpr float DEFAULT_SHARDSIZE = 10'000'000;

  explicit ShardedIndex(std::vector<std::shared_ptr<InvertedIndex>> indexes,
                        size_t max_shard_size = DEFAULT_SHARDSIZE)
      : _indexes(std::move(indexes)), _max_shard_size(max_shard_size) {}

  explicit ShardedIndex(size_t max_shard_size = DEFAULT_SHARDSIZE)
      : _max_shard_size(max_shard_size) {}

  explicit ShardedIndex(const ar::Archive& archive);

  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs);

  std::vector<DocScore> query(const std::string& query, uint32_t k,
                              bool parallelize = true) const;

  std::vector<DocScore> rank(const std::string& query,
                             const std::unordered_set<DocId>& candidates,
                             uint32_t k, bool parallelize = true) const;

  void remove(const std::vector<DocId>& ids);

  size_t size() const;

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<ShardedIndex> fromArchive(const ar::Archive& archive);

 private:
  std::vector<std::shared_ptr<InvertedIndex>> _indexes;
  size_t _max_shard_size;
};

}  // namespace thirdai::search
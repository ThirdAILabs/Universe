#pragma once

#include <search/src/neural_db/Chunk.h>
#include <search/src/neural_db/Constraints.h>

namespace thirdai::search::ndb {

class NeuralDB {
 public:
  virtual void insert(const std::string& document,
                      const std::vector<std::string>& chunks,
                      const std::vector<MetadataMap>& metadata,
                      const std::optional<std::string>& doc_id) = 0;

  virtual std::vector<std::pair<Chunk, float>> query(const std::string& query,
                                                     uint32_t top_k) = 0;

  virtual std::vector<std::pair<Chunk, float>> rank(
      const std::string& query, const QueryConstraints& constraints,
      uint32_t top_k) = 0;

  virtual void finetune(const std::vector<std::vector<ChunkId>>& chunk_ids,
                        const std::vector<std::string>& queries) = 0;

  virtual void deleteDoc(const DocId& doc, uint32_t version) = 0;

  virtual void prune() = 0;

  virtual ~NeuralDB() = default;
};

}  // namespace thirdai::search::ndb
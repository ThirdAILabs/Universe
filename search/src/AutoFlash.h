#pragma once

#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/MinHash.h>
#include <archive/src/Map.h>
#include <search/src/Flash.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace thirdai::search {

class AutoFlash {
 public:
  explicit AutoFlash(const std::string& dataset_size, TokenizerPtr tokenizer);

  AutoFlash(uint32_t hashes_per_table, uint32_t num_tables, uint32_t range,
            uint32_t reservoir_size, TokenizerPtr tokenizer);

  explicit AutoFlash(const ar::Archive& archive);

  void index(const std::vector<uint32_t>& ids,
             const std::vector<std::string>& docs);

  std::vector<std::vector<DocScore>> queryBatch(
      const std::vector<std::string>& queries, uint32_t k) const;

  std::vector<DocScore> query(const std::string& query, uint32_t k) const;

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<AutoFlash> fromArchive(const ar::Archive& archive);

  void save(const std::string& filename) const;

  void save_stream(std::ostream& ostream) const;

  static std::shared_ptr<AutoFlash> load(const std::string& filename);

  static std::shared_ptr<AutoFlash> load_stream(std::istream& istream);

 private:
  static Flash defaultFlash(const std::string& dataset_size);

  BoltVector tokenize(const std::string& string) const;

  Flash _flash;
  TokenizerPtr _tokenizer;
};

}  // namespace thirdai::search
#pragma once

#include <search/src/inverted_index/Tokenizer.h>
#include <search/src/neural_db/Chunk.h>
#include <unordered_map>

namespace thirdai::search::ndb {

using HashedToken = uint32_t;

using BatchTokens = std::unordered_map<HashedToken, std::vector<ChunkCount>>;

class TextProcessor {
 public:
  explicit TextProcessor(TokenizerPtr tokenizer)
      : _tokenizer(std::move(tokenizer)) {}

  std::pair<BatchTokens, std::vector<uint32_t>> process(
      ChunkId start_id, const std::vector<std::string>& chunks) const;

  std::vector<HashedToken> tokenize(const std::string& text) const;

 private:
  TokenizerPtr _tokenizer;
};

}  // namespace thirdai::search::ndb
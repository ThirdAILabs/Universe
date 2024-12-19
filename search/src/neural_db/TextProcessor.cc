#include "TextProcessor.h"
#include <dataset/src/utils/TokenEncoding.h>
#include <unordered_map>

namespace thirdai::search::ndb {

std::pair<BatchTokens, std::vector<uint32_t>> TextProcessor::process(
    ChunkId start_id, const std::vector<std::string>& chunks) const {
  BatchTokens batch_token_counts;

  std::vector<uint32_t> chunk_lens(chunks.size());

  for (size_t i = 0; i < chunks.size(); i++) {
    const auto tokens = tokenize(chunks.at(i));
    chunk_lens[i] = tokens.size();

    std::unordered_map<HashedToken, uint32_t> chunk_token_counts;
    for (const uint32_t token : tokens) {
      chunk_token_counts[token]++;
    }

    for (const auto& [token, count] : chunk_token_counts) {
      batch_token_counts[token].emplace_back(start_id + i, count);
    }
  }

  return std::make_pair(std::move(batch_token_counts), std::move(chunk_lens));
}

std::vector<HashedToken> TextProcessor::tokenize(
    const std::string& text) const {
  auto tokens = _tokenizer->tokenize(text);
  return dataset::token_encoding::hashTokens(tokens);
}

}  // namespace thirdai::search::ndb
#include "TextProcessor.h"
#include <dataset/src/utils/TokenEncoding.h>
#include <unordered_map>

namespace thirdai::search::ndb {

std::pair<BatchTokens, std::vector<uint32_t>> TextProcessor::process(
    ChunkId start_id, const std::vector<std::string>& chunks,
    const std::optional<std::string>& partition) const {
  BatchTokens batch_token_counts;

  std::vector<uint32_t> chunk_lens(chunks.size());

  for (size_t i = 0; i < chunks.size(); i++) {
    const auto tokens = tokenize(chunks.at(i), partition);
    chunk_lens[i] = tokens.size();

    std::unordered_map<std::string, uint32_t> chunk_token_counts;
    for (const auto& token : tokens) {
      chunk_token_counts[token]++;
    }

    for (const auto& [token, count] : chunk_token_counts) {
      batch_token_counts[token].emplace_back(start_id + i, count);
    }
  }

  return std::make_pair(std::move(batch_token_counts), std::move(chunk_lens));
}

std::vector<std::string> TextProcessor::tokenize(
    const std::string& text,
    const std::optional<std::string>& partition) const {
  auto tokens = _tokenizer->tokenize(text);
  auto hashes = dataset::token_encoding::hashTokens(tokens);
  if (!partition) {
    std::vector<std::string> serialized;
    serialized.reserve(hashes.size());
    for (const auto& hash : hashes) {
      std::string s;
      s.append(reinterpret_cast<const char*>(&hash), sizeof(HashedToken));
      serialized.push_back(s);
    }
    return serialized;
  }
  std::vector<std::string> serialized;
  serialized.reserve(hashes.size());
  for (const auto& hash : hashes) {
    std::string s;
    s.reserve(sizeof(HashedToken) + partition->size());
    s.append(partition->data(), partition->size());
    s.append(reinterpret_cast<const char*>(&hash), sizeof(HashedToken));
    serialized.push_back(s);
  }
  return serialized;
}

}  // namespace thirdai::search::ndb
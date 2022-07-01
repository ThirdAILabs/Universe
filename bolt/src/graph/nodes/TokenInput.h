#pragma once

#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <vector>

namespace thirdai::bolt {

/**
 * This class does not implement the Node interface because it is not
 * replaceable for other types of nodes like the regular Input node is. If a
 * node requires token input then it must take in a TokenInput node directly,
 * since no other node type will output tokens.
 */
class TokenInput {
 public:
  TokenInput() {}

  void setTokenInputs(dataset::BoltTokenBatch* tokens) { _tokens = tokens; }

  const std::vector<uint32_t>& getTokens(uint32_t batch_index) {
    return (*_tokens)[batch_index];
  }

 private:
  dataset::BoltTokenBatch* _tokens;
};

using TokenInputPtr = std::shared_ptr<TokenInput>;

}  // namespace thirdai::bolt
#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace thirdai::dataset {

class PairgramHasher {
 public:
  static bolt::BoltVector computePairgrams(std::string_view sentence,
                                           uint32_t output_range) {
    auto unigrams = computeUnigrams(sentence);
    return computePairgramsFromUnigrams(unigrams, output_range);
  }

  static bolt::BoltVector computePairgramsFromUnigrams(
      const std::vector<uint32_t>& unigram_hashes, uint32_t output_range) {
    std::unordered_map<uint32_t, uint32_t> pairgram_hashes;

    for (uint32_t token = 0; token < unigram_hashes.size(); token++) {
      for (uint32_t prev_token = 0; prev_token <= token; prev_token++) {
        uint32_t combined_hash = hashing::HashUtils::combineHashes(
            unigram_hashes[prev_token], unigram_hashes[token]);
        combined_hash = combined_hash % output_range;
        pairgram_hashes[combined_hash]++;
      }
    }

    bolt::BoltVector data_vec(pairgram_hashes.size(), false, false);
    uint32_t index = 0;
    for (auto& entry : pairgram_hashes) {
      data_vec.active_neurons[index] = entry.first;
      data_vec.activations[index] = entry.second;
      index++;
    }

    return data_vec;
  }

  static std::vector<uint32_t> computeUnigrams(std::string_view sentence) {
    std::vector<uint32_t> unigram_hashes;
    bool prev_is_space = true;
    uint32_t start_of_word_offset;
    for (uint32_t i = 0; i < sentence.size(); i++) {
      if (prev_is_space && !std::isspace(sentence[i])) {
        start_of_word_offset = i;
        prev_is_space = false;
      }
      if (!prev_is_space && std::isspace(sentence[i])) {
        uint32_t len = i - start_of_word_offset;

        uint32_t hash = hashing::MurmurHash(
            sentence.data() + start_of_word_offset, len, HASH_SEED);
        unigram_hashes.push_back(hash);
        prev_is_space = true;
      }
    }
    if (!prev_is_space) {
      uint32_t len = sentence.size() - start_of_word_offset;
      uint32_t hash = hashing::MurmurHash(
          sentence.data() + start_of_word_offset, len, HASH_SEED);
      unigram_hashes.push_back(hash);
    }

    return unigram_hashes;
  }

  static constexpr uint32_t HASH_SEED = 3829;
};

}  // namespace thirdai::dataset

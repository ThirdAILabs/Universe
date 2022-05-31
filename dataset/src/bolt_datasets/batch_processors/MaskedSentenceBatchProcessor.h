#pragma once

#include "PairgramHasher.h"
#include <bolt/src/layers/BoltVector.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/batch_types/MaskedSentenceBatch.h>
#include <dataset/src/bolt_datasets/BatchProcessor.h>
#include <random>

namespace thirdai::dataset {

class MaskedSentenceBatchProcessor final
    : public BatchProcessor<MaskedSentenceBatch> {
 public:
  explicit MaskedSentenceBatchProcessor(uint32_t output_range)
      : _output_range(output_range),
        _unknown_token_hash(
            hashing::MurmurHash("[UNK]", 5, PairgramHasher::HASH_SEED)),
        _rand(723204) {}

  std::optional<BoltDataLabelPair<MaskedSentenceBatch>> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<bolt::BoltVector> vectors(rows.size());
    std::vector<uint32_t> masked_indices(rows.size());
    std::vector<bolt::BoltVector> labels(rows.size());

    for (uint32_t i = 0; i < rows.size(); i++) {  // NOLINT
      auto [vec, index, label] = processRow(rows[i]);
      vectors.push_back(std::move(vec));
      masked_indices.push_back(index);
      labels.push_back(std::move(label));
    }

    return std::make_pair(
        MaskedSentenceBatch(std::move(vectors), std::move(masked_indices)),
        bolt::BoltBatch(std::move(labels)));
  }

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

 private:
  std::tuple<bolt::BoltVector, uint32_t, bolt::BoltVector> processRow(
      const std::string& row) {
    auto unigrams = PairgramHasher::computeUnigrams(row);

    uint32_t masked_index = _rand() % unigrams.size();

    uint32_t word_hash = unigrams[masked_index];
    unigrams[masked_index] = _unknown_token_hash;

    uint32_t word_id;
    if (_word_hashes_to_ids.count(word_hash)) {
      word_id = _word_hashes_to_ids.at(word_hash);
    } else {
      word_id = _word_hashes_to_ids.size();
      _word_hashes_to_ids[word_hash] = word_id;
    }

    bolt::BoltVector label(1, false, false);
    label.active_neurons[0] = word_id;
    label.activations[0] = 1.0;

    return {
        PairgramHasher::computePairgramsFromUnigrams(unigrams, _output_range),
        masked_index, std::move(label)};
  }

  std::unordered_map<uint32_t, uint32_t> _word_hashes_to_ids;
  uint32_t _output_range;
  uint32_t _unknown_token_hash;
  std::mt19937 _rand;
};

}  // namespace thirdai::dataset
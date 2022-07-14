#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/batch_types/MaskedSentenceBatch.h>
#include <dataset/src/bolt_datasets/BatchProcessor.h>
#include <dataset/src/encodings/text/TextEncodingUtils.h>
#include <random>
#include <unordered_map>

namespace thirdai::dataset {

class MaskedSentenceBatchProcessor final
    : public BatchProcessor<MaskedSentenceBatch> {
 public:
  explicit MaskedSentenceBatchProcessor(uint32_t output_range)
      : _output_range(output_range),
        _unknown_token_hash(TextEncodingUtils::computeUnigram("[UNK]", 5)),
        _rand(723204) {}

  std::optional<BoltDataLabelPair<MaskedSentenceBatch>> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<bolt::BoltVector> vectors(rows.size());
    std::vector<std::vector<uint32_t>> masked_indices(rows.size());
    std::vector<bolt::BoltVector> labels(rows.size());

#pragma omp parallel for default(none) \
    shared(rows, vectors, masked_indices, labels)
    for (uint32_t i = 0; i < rows.size(); i++) {
      auto [vec, index, label] = processRow(rows[i]);
      vectors[i] = std::move(vec);
      masked_indices[i] = {index};
      labels[i] = std::move(label);
    }

    return std::make_pair(
        MaskedSentenceBatch(std::move(vectors), std::move(masked_indices)),
        bolt::BoltBatch(std::move(labels)));
  }

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  const std::unordered_map<uint32_t, uint32_t>& getWordToIDMap() const {
    return _word_hashes_to_ids;
  }

 private:
  std::tuple<bolt::BoltVector, uint32_t, bolt::BoltVector> processRow(
      const std::string& row) {
    auto unigrams = TextEncodingUtils::computeRawUnigrams(row);

    uint32_t masked_index = _rand() % unigrams.size();

    uint32_t masked_word_hash = unigrams[masked_index];
    unigrams[masked_index] = _unknown_token_hash;

    // We are using the hash of the masked word to find it's ID because the
    // chance that two words have the same hash in the range [0, 2^32) are very
    // small, and by using this hash we avoid having to store all of the words
    // in the sentence and we can simply do a single pass over it and compute
    // the hashes.
    uint32_t word_id;
#pragma omp critical
    {
      if (_word_hashes_to_ids.count(masked_word_hash)) {
        word_id = _word_hashes_to_ids.at(masked_word_hash);
      } else {
        word_id = _word_hashes_to_ids.size();
        _word_hashes_to_ids[masked_word_hash] = word_id;
      }
    }

    bolt::BoltVector label(1, false, false);
    label.active_neurons[0] = word_id;
    label.activations[0] = 1.0;

    return {TextEncodingUtils::computePairgramsFromUnigrams(unigrams,
                                                            _output_range),
            masked_index, std::move(label)};
  }

  std::unordered_map<uint32_t, uint32_t> _word_hashes_to_ids;
  uint32_t _output_range;
  uint32_t _unknown_token_hash;
  std::mt19937 _rand;
};

}  // namespace thirdai::dataset
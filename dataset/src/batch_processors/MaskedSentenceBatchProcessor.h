#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/encodings/text/TextEncodingUtils.h>
#include <optional>
#include <random>
#include <unordered_map>

namespace thirdai::dataset {

class MaskedSentenceBatchProcessor final
    : public BatchProcessor<BoltBatch, BoltBatch, BoltBatch> {
 public:
  explicit MaskedSentenceBatchProcessor(uint32_t output_range)
      : _output_range(output_range),
        _unknown_token_hash(TextEncodingUtils::computeUnigram(
            /* key= */ "[UNK]", /* len= */ 5)),
        _rand(723204),
        _masked_tokens_percentage(std::nullopt) {}

  explicit MaskedSentenceBatchProcessor(uint32_t output_range,
                                        const float masked_tokens_percentage)
      : MaskedSentenceBatchProcessor(output_range) {
    _masked_tokens_percentage = masked_tokens_percentage;
  }

  std::tuple<BoltBatch, BoltBatch, BoltBatch> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<BoltVector> vectors(rows.size());
    std::vector<BoltVector> masked_indices(rows.size());
    std::vector<BoltVector> labels(rows.size());

#pragma omp parallel for default(none) \
    shared(vectors, masked_indices, labels, rows)
    for (uint32_t i = 0; i < rows.size(); i++) {
      auto [row_pairgrams, indices, label] = processRow(rows[i]);
      vectors[i] = std::move(row_pairgrams);
      masked_indices[i] = indices;
      labels[i] = std::move(label);
    }

    return std::make_tuple(BoltBatch(std::move(vectors)),
                           BoltBatch(std::move(masked_indices)),
                           BoltBatch(std::move(labels)));
  }

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  const std::unordered_map<uint32_t, uint32_t>& getWordToIDMap() const {
    return _word_hashes_to_ids;
  }

 private:
  std::tuple<BoltVector, BoltVector, BoltVector> processRow(
      const std::string& row) {
    auto unigrams = TextEncodingUtils::computeRawUnigrams(row);

    uint32_t size = unigrams.size();
    std::vector<uint32_t> masked_indices;
    std::vector<uint32_t> masked_word_hashes;

    uint32_t masked_tokens_size = 1;

    if (_masked_tokens_percentage.has_value()) {
      masked_tokens_size =
          static_cast<uint32_t>(size * _masked_tokens_percentage.value());
      for (uint32_t unigram_index = 0; unigram_index < masked_tokens_size;
           unigram_index++) {
        uint32_t masked_index = _rand() % size;
        masked_indices.push_back(masked_index);
        masked_word_hashes.push_back(unigrams[masked_index]);
        unigrams[masked_index] = _unknown_token_hash;
      }
    } else {
      uint32_t masked_index = _rand() % size;
      masked_indices.push_back(masked_index);
      masked_word_hashes.push_back(unigrams[masked_index]);
      unigrams[masked_indices[0]] = _unknown_token_hash;
    }

    // We are using the hash of the masked word to find its ID because the
    // chance that two words have the same hash in the range [0, 2^32) are very
    // small, and by using this hash we avoid having to store all of the words
    // in the sentence and we can simply do a single pass over it and compute
    // the hashes.
    std::vector<uint32_t> masked_word_ids;

#pragma omp critical
    {
      for (uint32_t masked_word_hash : masked_word_hashes) {
        if (_word_hashes_to_ids.count(masked_word_hash)) {
          masked_word_ids.push_back(_word_hashes_to_ids.at(masked_word_hash));
        } else {
          auto map_size = _word_hashes_to_ids.size();
          masked_word_ids.push_back(map_size);

          _word_hashes_to_ids[masked_word_hash] = map_size;
        }
      }
    }

    int len = _masked_tokens_percentage.has_value() ? masked_tokens_size : 1;

    BoltVector label(len, false, false);

    for (uint32_t index = 0; index < masked_tokens_size; index++) {
      label.active_neurons[index] = masked_word_ids[index];
      label.activations[index] = 1.0;
    }

    const std::vector<float> values(masked_tokens_size, 1.0);
    auto pairgrams = TextEncodingUtils::computePairgramsFromUnigrams(
        unigrams, _output_range);

    return {pairgrams, BoltVector::makeSparseVector(masked_indices, values),
            std::move(label)};
  }

  std::unordered_map<uint32_t, uint32_t> _word_hashes_to_ids;
  uint32_t _output_range;
  uint32_t _unknown_token_hash;
  std::mt19937 _rand;

  // This represents the percentage of tokens masked in any input sequence.
  // For instance, if _masked_tokens_percentage = 0.10, then 10% of the
  // words in the input sequence are randomly masked.
  std::optional<float> _masked_tokens_percentage;
};  // namespace thirdai::dataset

}  // namespace thirdai::dataset
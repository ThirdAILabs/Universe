#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/MurmurHash.h>
#include <_types/_uint32_t.h>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::dataset {

class MaskedSentenceBatchProcessor final
    : public BatchProcessor<BoltBatch, BoltBatch, BoltBatch> {
 public:
  explicit MaskedSentenceBatchProcessor(uint32_t output_range)
      : _output_range(output_range),
        _masked_token_hash(TextEncodingUtils::computeUnigram(
            /* key= */ "[MASK]", /* len= */ 6)),
        _rand(723204),
        _masked_tokens_percentage(std::nullopt) {}

  MaskedSentenceBatchProcessor(uint32_t output_range,
                               const float masked_tokens_percentage)
      : MaskedSentenceBatchProcessor(output_range) {
    _masked_tokens_percentage = masked_tokens_percentage;
    _unknown_token_hash = TextEncodingUtils::computeUnigram(  // NOLINT
        /* key= */ "[UNK]", /* len= */ 5);
  }

  std::tuple<BoltBatch, BoltBatch, BoltBatch> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<BoltVector> vectors;
    std::vector<BoltVector> masked_indices;
    std::vector<BoltVector> labels;

#pragma omp parallel for default(none) \
    shared(rows, vectors, masked_indices, labels)
    for (const auto& row : rows) {
      auto unigram_vector = TextEncodingUtils::computeRawUnigrams(row);
      auto [row_pairgrams, indices, row_labels] = processRow(unigram_vector);

      vectors.reserve(vectors.size() + row_pairgrams.size());
      labels.reserve(labels.size() + row_labels.size());

      vectors.insert(vectors.end(), row_pairgrams.begin(), row_pairgrams.end());
      masked_indices.push_back(std::move(indices));

      labels.insert(labels.end(), row_labels.begin(), row_labels.end());
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
  using PairgramsMaskedIndicesLabelsTuple =
      std::tuple<std::vector<BoltVector>, BoltVector, std::vector<BoltVector>>;

  PairgramsMaskedIndicesLabelsTuple processRow(
      const std::vector<uint32_t>& unigrams) {
    uint32_t size = unigrams.size();
    std::vector<uint32_t> masked_indices;
    std::vector<uint32_t> masked_word_hashes;

    uint32_t masked_tokens_size =
        (_masked_tokens_percentage.has_value())
            ? static_cast<uint32_t>(size * _masked_tokens_percentage.value())
            : 1;

    std::vector<std::vector<uint32_t>> unigram_copies(masked_tokens_size,
                                                      unigrams);
    std::unordered_set<uint32_t> already_masked_tokens;

    for (uint32_t index = 0; index < masked_tokens_size; index++) {
      uint32_t masked_index = _rand() % size;
      if (already_masked_tokens.count(masked_index)) {
        continue;
      }
      masked_indices.push_back(masked_index);
      masked_word_hashes.push_back(unigrams[masked_index]);
      unigram_copies[index][masked_index] = _masked_token_hash;
    }

    for (auto unigram_copy : unigram_copies) {
      for (uint32_t masked_index : masked_indices) {
        if (unigram_copy[masked_index] == _masked_token_hash) {
          continue;
        }
        unigram_copy[masked_index] = _unknown_token_hash;
      }
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
          uint32_t map_size = _word_hashes_to_ids.size();
          masked_word_ids.push_back(map_size);

          _word_hashes_to_ids[masked_word_hash] = map_size;
        }
      }
    }

    std::vector<BoltVector> labels;
    std::for_each(masked_word_ids.begin(), masked_word_ids.end(),
                  [&labels](uint32_t masked_word_id) {
                    labels.push_back(
                        BoltVector::makeSparseVector({masked_word_id}, {1.0}));
                  });

    std::vector<BoltVector> pairgrams;
    std::for_each(
        unigram_copies.begin(), unigram_copies.end(),
        [&pairgrams, this](std::vector<uint32_t> unigram_copy) {
          pairgrams.push_back(TextEncodingUtils::computePairgramsFromUnigrams(
              unigram_copy, _output_range));
        });

    return {std::move(pairgrams),
            BoltVector::makeSparseVector(
                masked_indices, std::vector<float>(masked_tokens_size, 1.0)),
            std::move(labels)};
  }

  std::unordered_map<uint32_t, uint32_t> _word_hashes_to_ids;
  uint32_t _output_range;
  uint32_t _unknown_token_hash;
  uint32_t _masked_token_hash;
  std::mt19937 _rand;

  // Represents the percentage of tokens masked in any input sequence.
  // For instance, if _masked_tokens_percentage = 0.10, then 10% of the
  // words in the input sequence are randomly masked.
  std::optional<float> _masked_tokens_percentage;
};  // namespace thirdai::dataset

}  // namespace thirdai::dataset
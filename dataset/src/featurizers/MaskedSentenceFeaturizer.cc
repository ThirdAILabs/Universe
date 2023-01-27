#include "MaskedSentenceFeaturizer.h"
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/Vocabulary.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::dataset {

std::vector<std::vector<BoltVector>> MaskedSentenceFeaturizer::featurize(
    const std::vector<std::string>& rows) {
  std::vector<BoltVector> vectors(rows.size());
  std::vector<BoltVector> masked_indices(rows.size());
  std::vector<BoltVector> labels(rows.size());

#pragma omp parallel for default(none) \
    shared(rows, vectors, masked_indices, labels)
  for (uint32_t i = 0; i < rows.size(); i++) {
    auto [row_pairgrams, indices, label] = processRow(rows[i]);
    vectors[i] = std::move(row_pairgrams);
    masked_indices[i] = std::move(indices);
    labels[i] = std::move(label);
  }

  return {std::move(vectors), std::move(masked_indices), std::move(labels)};
}

std::tuple<BoltVector, BoltVector, BoltVector>
MaskedSentenceFeaturizer::processRow(const std::string& row) {
  std::vector<uint32_t> unigrams = _vocab->encode(row);

  uint32_t size = unigrams.size();
  std::vector<uint32_t> masked_indices;
  std::vector<uint32_t> masked_word_ids;

  uint32_t masked_tokens_size =
      (_masked_tokens_percentage.has_value())
          ? static_cast<uint32_t>(size * _masked_tokens_percentage.value())
          : 1;
  std::unordered_set<uint32_t> already_masked_tokens;
  uint32_t unigram_index = 0;

  while (unigram_index < masked_tokens_size) {
    uint32_t masked_index = _rand() % size;
    if (already_masked_tokens.count(masked_index)) {
      continue;
    }
    masked_indices.push_back(masked_index);
    already_masked_tokens.insert(masked_index);
    masked_word_ids.push_back(unigrams[masked_index]);
    unigrams[masked_index] = _vocab->maskId();

    unigram_index++;
  }

  BoltVector label = BoltVector::makeSparseVector(
      masked_word_ids, std::vector<float>(masked_word_ids.size(), 1.0));

  auto pairgrams =
      TokenEncoding::computePairgramsFromUnigrams(unigrams, _output_range);

  return {std::move(pairgrams),
          BoltVector::makeSparseVector(
              masked_indices, std::vector<float>(masked_tokens_size, 1.0)),
          std::move(label)};
}

}  // namespace thirdai::dataset

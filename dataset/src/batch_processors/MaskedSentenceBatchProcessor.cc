#include "dataset/src/batch_processors/MaskedSentenceBatchProcessor.h"
#include <vector>

namespace thirdai::dataset {

namespace detail {
// Sample mask positions using generator's randomness. Returns
// masked_tokens_size positions in size. Sampling without replacement.
std::vector<uint32_t> generate_mask_positions(std::mt19937& generator,
                                              uint32_t masked_tokens_size,
                                              uint32_t size) {
  std::vector<uint32_t> masked_indices;

  std::unordered_set<uint32_t> already_masked_tokens;

  while (masked_indices.size() < masked_tokens_size) {
    uint32_t masked_index = generator() % size;
    if (!already_masked_tokens.count(masked_index)) {
      already_masked_tokens.insert(masked_index);
      masked_indices.push_back(masked_index);
    }
  }
  return masked_indices;
}

std::tuple<BoltVector, BoltVector, BoltVector> masked_sample(
    std::vector<uint32_t> unigrams, const std::vector<uint32_t>& mask_indices,
    uint32_t maskId, uint32_t output_range) {
  // Mask unigrams, collect the mask unigram ids for labels.
  std::vector<uint32_t> mask_unigram_ids;
  for (auto mask_index : mask_indices) {
    mask_unigram_ids.push_back(unigrams[mask_index]);
    unigrams[mask_index] = maskId;
  }

  // Construct pairgrams from unigrams
  BoltVector pairgrams =
      TextEncodingUtils::computePairgramsFromUnigrams(unigrams, output_range);

  BoltVector mask_info = BoltVector::makeSparseVector(
      mask_indices, std::vector<float>(mask_indices.size(), 1.0));

  BoltVector label = BoltVector::makeSparseVector(
      mask_unigram_ids, std::vector<float>(mask_unigram_ids.size(), 1.0));

  return {std::move(pairgrams), std::move(mask_info), std::move(label)};
}

}  // namespace detail

// Sample mask positions using generator's randomness. Returns
// masked_tokens_size positions in size. Sampling without replacement.
inline std::vector<uint32_t> generate_mask_positions(
    std::mt19937& generator, uint32_t masked_tokens_size, uint32_t size) {
  std::vector<uint32_t> masked_indices;

  std::unordered_set<uint32_t> already_masked_tokens;

  while (masked_indices.size() < masked_tokens_size) {
    uint32_t masked_index = generator() % size;
    if (!already_masked_tokens.count(masked_index)) {
      already_masked_tokens.insert(masked_index);
      masked_indices.push_back(masked_index);
    }
  }
  return masked_indices;
}

inline std::tuple<BoltVector, BoltVector, BoltVector> sample(
    std::vector<uint32_t>& unigrams, const std::vector<uint32_t>& mask_indices,
    uint32_t maskId, uint32_t output_range) {
  // Mask unigrams, collect the mask unigram ids for labels.
  std::vector<uint32_t> mask_unigram_ids;
  for (auto mask_index : mask_indices) {
    mask_unigram_ids.push_back(unigrams[mask_index]);
    unigrams[mask_index] = maskId;
  }

  // Construct pairgrams from unigrams
  BoltVector pairgrams =
      TextEncodingUtils::computePairgramsFromUnigrams(unigrams, output_range);

  BoltVector mask_info = BoltVector::makeSparseVector(
      mask_indices, std::vector<float>(mask_indices.size(), 1.0));

  BoltVector label = BoltVector::makeSparseVector(
      mask_unigram_ids, std::vector<float>(mask_unigram_ids.size(), 1.0));

  return {std::move(pairgrams), std::move(mask_info), std::move(label)};
}

MaskedSentenceBatchProcessor::MaskedSentenceBatchProcessor(
    std::shared_ptr<Vocabulary> vocab, uint32_t output_range)
    : _vocab(std::move(vocab)),
      _output_range(output_range),
      _rand(723204),
      _masked_tokens_percentage(std::nullopt) {}

MaskedSentenceBatchProcessor::MaskedSentenceBatchProcessor(
    std::shared_ptr<Vocabulary> vocab, uint32_t output_range,
    float masked_tokens_percentage)
    : MaskedSentenceBatchProcessor(std::move(vocab), output_range) {
  _masked_tokens_percentage = masked_tokens_percentage;
}

std::tuple<BoltBatch, BoltBatch, BoltBatch>
MaskedSentenceBatchProcessor::createBatch(
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

  return std::make_tuple(BoltBatch(std::move(vectors)),
                         BoltBatch(std::move(masked_indices)),
                         BoltBatch(std::move(labels)));
}

bool MaskedSentenceBatchProcessor::expectsHeader() const { return false; }

void MaskedSentenceBatchProcessor::processHeader(const std::string& header) {
  (void)header;
}

std::tuple<BoltVector, BoltVector, BoltVector>
MaskedSentenceBatchProcessor::processRow(const std::string& row) {
  std::vector<uint32_t> unigrams = _vocab->encode(row);

  uint32_t masked_tokens_size =
      (_masked_tokens_percentage.has_value())
          ? static_cast<uint32_t>(unigrams.size() *
                                  _masked_tokens_percentage.value())
          : 1;

  std::vector<uint32_t> mask_indices = detail::generate_mask_positions(
      _rand, masked_tokens_size, unigrams.size());

  return detail::masked_sample(std::move(unigrams), mask_indices,
                               _vocab->maskId(), _output_range);
}

std::tuple<BoltBatch, BoltBatch, BoltBatch> inferenceBatch(
    std::shared_ptr<Vocabulary> vocab, const std::vector<std::string>& rows,
    uint32_t output_range, float mask_percentage /*=0.0f*/) {
  MaskedSentenceBatchProcessor batch_processor(std::move(vocab), output_range,
                                               mask_percentage);
  return batch_processor.createBatch(rows);
}

std::tuple<BoltVector, BoltVector, BoltVector> inferenceSample(
    const std::shared_ptr<Vocabulary>& vocab, const std::string& row,
    const std::vector<uint32_t>& mask_indices, uint32_t output_range) {
  std::vector<uint32_t> unigrams = vocab->encode(row);

  return detail::masked_sample(std::move(unigrams), mask_indices,
                               vocab->maskId(), output_range);
}

}  // namespace thirdai::dataset

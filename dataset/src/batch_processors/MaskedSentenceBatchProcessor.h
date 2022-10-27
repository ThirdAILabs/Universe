#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/Vocabulary.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::dataset {

// The following function is less-than ideal. But we will make do with this in
// the short-term.
std::tuple<BoltBatch, BoltBatch, BoltBatch> inferenceBatch(
    std::shared_ptr<Vocabulary> vocab, const std::vector<std::string>& rows,
    uint32_t output_range, float mask_percentage = 0.0);

// This is a bad API, we need a unigrams -> masked_indices level API. As of now
// the client has no way of knowing row has masked_indices.
std::tuple<BoltVector, BoltVector, BoltVector> inferenceSample(
    const std::shared_ptr<Vocabulary>& vocab, const std::string& row,
    const std::vector<uint32_t>& mask_indices, uint32_t output_range);

class MaskedSentenceBatchProcessor final
    : public BatchProcessor<BoltBatch, BoltBatch, BoltBatch> {
 public:
  MaskedSentenceBatchProcessor(std::shared_ptr<Vocabulary> vocab,
                               uint32_t output_range);

  MaskedSentenceBatchProcessor(std::shared_ptr<Vocabulary> vocab,
                               uint32_t output_range,
                               float masked_tokens_percentage);

  std::tuple<BoltBatch, BoltBatch, BoltBatch> createBatch(
      const std::vector<std::string>& rows) final;

  bool expectsHeader() const final;

  void processHeader(const std::string& header) final;

  std::tuple<BoltVector, BoltVector, BoltVector> processRow(
      const std::string& row);

 private:
  std::shared_ptr<Vocabulary> _vocab;

  uint32_t _output_range;
  std::mt19937 _rand;

  // Represents the percentage of tokens masked in any input sequence.
  // For instance, if _masked_tokens_percentage = 0.10, then 10% of the
  // words in the input sequence are randomly masked.
  std::optional<float> _masked_tokens_percentage;
};  // namespace thirdai::dataset

}  // namespace thirdai::dataset

#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>
#include <dataset/src/Vocabulary.h>
#include <random>

namespace thirdai::dataset {

class MaskedSentenceFeaturizer final : public Featurizer {
 public:
  explicit MaskedSentenceFeaturizer(std::shared_ptr<Vocabulary> vocab,
                                    uint32_t output_range)
      : _vocab(std::move(vocab)),
        _output_range(output_range),
        _rand(723204),
        _masked_tokens_percentage(std::nullopt) {}

  MaskedSentenceFeaturizer(std::shared_ptr<Vocabulary> vocab,
                           uint32_t output_range,
                           const float masked_tokens_percentage)
      : MaskedSentenceFeaturizer(std::move(vocab), output_range) {
    _masked_tokens_percentage = masked_tokens_percentage;
  }

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) final;

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  size_t getNumDatasets() final { return 3; }

 private:
  std::tuple<BoltVector, BoltVector, BoltVector> processRow(
      const std::string& row);

  std::shared_ptr<Vocabulary> _vocab;

  uint32_t _output_range;
  std::mt19937 _rand;

  // Represents the percentage of tokens masked in any input sequence.
  // For instance, if _masked_tokens_percentage = 0.10, then 10% of the
  // words in the input sequence are randomly masked.
  std::optional<float> _masked_tokens_percentage;
};  // namespace thirdai::dataset

using MaskedSentenceFeaturizerPtr = std::shared_ptr<MaskedSentenceFeaturizer>;

}  // namespace thirdai::dataset

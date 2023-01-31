#include "TextGenerationFeaturizer.h"
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/HashUtils.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <algorithm>
#include <cctype>
#include <iterator>

namespace thirdai::dataset {

TextGenerationFeaturizer::TextGenerationFeaturizer(uint32_t sequence_len,
                                                   uint32_t vocab_size)
    : _sequence_len(sequence_len), _vocab_size(vocab_size) {}

std::vector<std::vector<BoltVector>> TextGenerationFeaturizer::featurize(
    const std::vector<std::string>& lines) {
  std::vector<std::pair<std::vector<BoltVector>, std::vector<BoltVector>>>
      featurized_samples(lines.size());

  // #pragma omp parallel for default(none) shared(lines, featurized_samples)
  for (uint32_t i = 0; i < lines.size(); i++) {
    featurized_samples[i] = featurizeText(lines[i]);
  }

  std::vector<BoltVector> vectors;
  std::vector<BoltVector> labels;

  for (auto& [vec, label] : featurized_samples) {
    vectors.insert(vectors.end(), std::make_move_iterator(vec.begin()),
                   std::make_move_iterator(vec.end()));
    labels.insert(labels.end(), std::make_move_iterator(label.begin()),
                  std::make_move_iterator(label.end()));
  }

  return {std::move(vectors), std::move(labels)};
}

std::pair<std::vector<BoltVector>, std::vector<BoltVector>>
TextGenerationFeaturizer::featurizeText(const std::string& line) const {
  std::vector<uint32_t> tokens = parseTokens(line);

  std::vector<BoltVector> vectors;
  std::vector<BoltVector> labels;

  for (uint32_t i = _sequence_len; i < tokens.size(); i++) {
    const uint32_t* phrase_start = tokens.data() + i - _sequence_len;
    auto pairgrams = token_encoding::pairgrams(phrase_start, _sequence_len);

    BoltVector vector(/* l= */ pairgrams.size(), /* is_dense= */ false,
                      /* has_gradient= */ false);
    std::copy(pairgrams.begin(), pairgrams.end(), vector.active_neurons);
    std::fill_n(vector.activations, vector.len, 1.0);

    BoltVector label = BoltVector::singleElementSparseVector(tokens[i]);

    vectors.emplace_back(std::move(vector));
    labels.emplace_back(std::move(label));
  }

  return std::make_pair(std::move(vectors), std::move(labels));
}

std::vector<uint32_t> TextGenerationFeaturizer::parseTokens(
    const std::string& line) {
  std::vector<uint32_t> tokens;

  const char* start = line.data();
  const char* line_end = line.data() + line.size();

  while (start != line_end) {
    char* end;
    tokens.push_back(std::strtoul(start, &end, /* base= */ 10));
    start = end;
  }

  return tokens;
}

}  // namespace thirdai::dataset

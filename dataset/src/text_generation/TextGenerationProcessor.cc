#include "TextGenerationProcessor.h"
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/HashUtils.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <json/include/nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <iterator>

using nlohmann::json;

namespace thirdai::dataset {

TextGenerationProcessor::TextGenerationProcessor(uint32_t seq_len,
                                                 uint32_t input_dim,
                                                 uint32_t output_dim)
    : _seq_len(seq_len), _input_dim(input_dim), _output_dim(output_dim) {}

std::pair<std::vector<BoltVector>, std::vector<BoltVector>>
TextGenerationProcessor::featurize(
    const std::vector<std::string>& lines) const {
  std::vector<std::pair<std::vector<BoltVector>, std::vector<BoltVector>>>
      featurized_samples(lines.size());

#pragma omp parallel for default(none) shared(lines, featurized_samples)
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

  return std::make_pair(std::move(vectors), std::move(labels));
}

std::pair<std::vector<BoltVector>, std::vector<BoltVector>>
TextGenerationProcessor::featurizeText(const std::string& line) const {
  auto doc = json::parse(line);

  std::string text = removePunctuationAndSpacing(doc["text"]);

  std::vector<uint32_t> tokens = token_encoding::unigrams(text);

  std::vector<BoltVector> vectors;
  std::vector<BoltVector> labels;

  for (uint32_t i = _seq_len; i < tokens.size(); i++) {
    auto pairgrams = token_encoding::pairgrams(std::vector<uint32_t>(
        tokens.begin() + i - _seq_len, tokens.begin() + i));
    token_encoding::mod(pairgrams, _input_dim);

    BoltVector vector(/* l= */ pairgrams.size(), /* is_dense= */ false,
                      /* has_gradient= */ false);
    std::copy(pairgrams.begin(), pairgrams.end(), vector.active_neurons);
    std::fill_n(vector.activations, vector.len, 1.0);

    uint32_t label_1 = tokens[i];
    uint32_t label_2 = hashing::simpleIntegerHash(label_1);

    BoltVector label(/* l= */ 2, /* is_dense= */ false,
                     /* has_gradient= */ false);
    label.active_neurons[0] = label_1 % _output_dim;
    label.active_neurons[1] = label_2 % _output_dim;
    label.activations[0] = 1.0;
    label.activations[1] = 1.0;

    vectors.emplace_back(std::move(vector));
    labels.emplace_back(std::move(label));
  }

  return std::make_pair(std::move(vectors), std::move(labels));
}

std::string TextGenerationProcessor::removePunctuationAndSpacing(
    const std::string& str) {
  std::string new_str;
  new_str.reserve(str.size());

  for (char c : str) {
    if (std::isspace(c)) {
      new_str.push_back(' ');
    } else if (!std::ispunct(c)) {
      new_str.push_back(c);
    }
  }
  return new_str;
}

}  // namespace thirdai::dataset

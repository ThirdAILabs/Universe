#include "TextGenerationFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/HashUtils.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <algorithm>
#include <cctype>
#include <iterator>
#include <stdexcept>

namespace thirdai::dataset {

TextGenerationFeaturizer::TextGenerationFeaturizer(uint32_t sequence_len,
                                                   uint32_t vocab_size,
                                                   uint32_t last_n_tokens,
                                                   bool pairgrams)
    : _sequence_len(sequence_len),
      _vocab_size(vocab_size),
      _last_n_tokens(last_n_tokens),
      _pairgrams(pairgrams) {
  if (_last_n_tokens >= sequence_len) {
    throw std::invalid_argument(
        "Last n tokens must be less than the sequence length.");
  }
}

std::vector<std::vector<BoltVector>> TextGenerationFeaturizer::featurize(
    const std::vector<std::string>& lines) {
  std::vector<std::vector<std::vector<BoltVector>>> featurized_samples(
      lines.size());

#pragma omp parallel for default(none) shared(lines, featurized_samples)
  for (uint32_t i = 0; i < lines.size(); i++) {
    featurized_samples[i] = featurizeText(lines[i]);
  }

  std::vector<std::vector<BoltVector>> data(3);

  for (auto& vectors : featurized_samples) {
    for (auto& sample : vectors) {
      for (uint32_t i = 0; i < sample.size(); i++) {
        data.at(i).push_back(std::move(sample[i]));
      }
    }
  }

  return data;
}

std::vector<std::vector<BoltVector>> TextGenerationFeaturizer::featurizeText(
    const std::string& line) const {
  std::vector<uint32_t> tokens = parseTokens(line);

  std::vector<std::vector<BoltVector>> vectors;

  for (uint32_t i = _sequence_len; i < tokens.size(); i++) {
    const uint32_t* phrase_start = tokens.data() + i - _sequence_len;

    std::vector<uint32_t> context_tokens;
    if (_pairgrams) {
      context_tokens = token_encoding::pairgrams(phrase_start, _sequence_len);
    } else {
      context_tokens =
          std::vector<uint32_t>(phrase_start, phrase_start + _sequence_len);
    }

    BoltVector vector(/* l= */ context_tokens.size(), /* is_dense= */ false,
                      /* has_gradient= */ false);
    std::copy(context_tokens.begin(), context_tokens.end(),
              vector.active_neurons);
    std::fill_n(vector.activations, vector.len, 1.0);

    BoltVector last_tokens(/* l= */ _last_n_tokens, /* is_dense= */ false,
                           /* has_gradient= */ false);
    for (uint32_t t = 0; t < _last_n_tokens; t++) {
      last_tokens.active_neurons[t] = tokens[i - _last_n_tokens + t];
      last_tokens.activations[t] = 1.0;
    }

    BoltVector label = BoltVector::singleElementSparseVector(tokens[i]);

    vectors.push_back(
        {std::move(vector), std::move(last_tokens), std::move(label)});
  }

  return vectors;
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

std::vector<BoltVector> TextGenerationFeaturizer::featurizeInferenceSample(
    const std::vector<uint32_t>& tokens) const {
  auto context = _pairgrams ? token_encoding::pairgrams(tokens) : tokens;

  BoltVector vector = BoltVector::makeSparseVector(
      context, std::vector<float>(context.size(), 1.0));

  BoltVector last_tokens(/* l= */ _last_n_tokens, /* is_dense= */ false,
                         /* has_gradient= */ false);
  for (uint32_t t = 0; t < _last_n_tokens; t++) {
    last_tokens.active_neurons[t] = tokens[tokens.size() - _last_n_tokens + t];
    last_tokens.activations[t] = 1.0;
  }

  return {std::move(vector), std::move(last_tokens)};
}

void TextGenerationFeaturizer::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void TextGenerationFeaturizer::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

TextGenerationFeaturizerPtr TextGenerationFeaturizer::load(
    const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

TextGenerationFeaturizerPtr TextGenerationFeaturizer::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<TextGenerationFeaturizer> deserialize_into(
      new TextGenerationFeaturizer());
  iarchive(*deserialize_into);
  return deserialize_into;
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TextGenerationFeaturizer)
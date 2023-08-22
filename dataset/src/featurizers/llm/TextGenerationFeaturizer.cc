#include "TextGenerationFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/HashUtils.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <algorithm>
#include <cctype>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

std::vector<std::vector<BoltVector>> TextGenerationFeaturizer::featurize(
    const std::vector<std::string>& lines) {
  std::vector<std::vector<std::vector<BoltVector>>> featurized_samples(
      lines.size());

#pragma omp parallel for default(none) shared(lines, featurized_samples)
  for (uint32_t i = 0; i < lines.size(); i++) {
    featurized_samples[i] = featurizeText(lines[i]);
  }

  std::vector<std::vector<BoltVector>> data(5);

  for (auto& vectors : featurized_samples) {
    for (auto& sample : vectors) {
      for (uint32_t i = 0; i < sample.size(); i++) {
        data.at(i).push_back(std::move(sample[i]));
      }
    }
  }

  return data;
}

static std::string getStringField(const json& json_object,
                                  const std::string& name) {
  if (!json_object[name].is_string()) {
    throw std::invalid_argument("Expected field '" + name +
                                "' to be a string.");
  }
  return json_object[name].get<std::string>();
}

std::vector<std::vector<BoltVector>> TextGenerationFeaturizer::featurizeText(
    const std::string& line) const {
  auto line_content = json::parse(line);
  if (!line_content.is_object()) {
    throw std::invalid_argument("Expected line to be a json object.");
  }

  auto tokens = getAllTokens(line_content);

  BoltVector prompt = promptContext(getPrompt(line_content));

  std::vector<std::vector<BoltVector>> vectors;

  size_t chunk_size = _context_featurizer.contextSize() + 1;

  for (size_t chunk_start = 0; chunk_start < tokens.size();
       chunk_start += chunk_size) {
    size_t chunk_end = std::min(tokens.size(), chunk_start + chunk_size);

    for (size_t i = chunk_start + 1; i < chunk_end; i++) {
      BoltVector label = BoltVector::singleElementSparseVector(tokens[i]);
      vectors.push_back({prompt,
                         _context_featurizer.lrcContext(tokens, chunk_start, i),
                         _context_featurizer.ircContext(tokens, chunk_start, i),
                         _context_featurizer.srcContext(tokens, chunk_start, i),
                         std::move(label)});
    }
  }

  return vectors;
}

BoltVector TextGenerationFeaturizer::promptContext(
    const std::vector<uint32_t>& prompt_tokens) {
  if (prompt_tokens.empty()) {
    // We use a single padding token if the prompt is empty to avoid passing
    // empty vectors into bolt.
    return BoltVector::singleElementSparseVector(0);
  }
  BoltVector prompt(/* l= */ prompt_tokens.size(), /* is_dense= */ false,
                    /* has_gradient= */ false);
  std::copy(prompt_tokens.begin(), prompt_tokens.end(), prompt.active_neurons);
  std::fill_n(prompt.activations, prompt.len, 1.0);
  return prompt;
}

std::vector<BoltVector> TextGenerationFeaturizer::featurizeInferenceSample(
    const std::vector<uint32_t>& prompt,
    const std::vector<uint32_t>& context) const {
  return {promptContext(prompt), _context_featurizer.lrcContext(context),
          _context_featurizer.ircContext(context),
          _context_featurizer.srcContext(context)};
}

std::vector<uint32_t> TextGenerationFeaturizer::getAllTokens(
    const json& line_content) {
  if (!line_content.contains("target")) {
    throw std::invalid_argument("Expected field 'target' in json object'");
  }

  std::vector<uint32_t> target_tokens =
      token_encoding::tokenIds(getStringField(line_content, "target"));

  return target_tokens;
}

std::vector<uint32_t> TextGenerationFeaturizer::getPrompt(
    const json& line_content) {
  if (line_content.contains("prompt")) {
    return token_encoding::tokenIds(getStringField(line_content, "prompt"));
  }
  return {};
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
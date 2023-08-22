#include "ContextAwareTextFeaturizer.h"
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

std::vector<std::vector<BoltVector>> ContextAwareTextFeaturizer::featurize(
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

std::vector<std::vector<BoltVector>> ContextAwareTextFeaturizer::featurizeText(
    const std::string& line) const {
  auto line_content = json::parse(line);
  if (!line_content.is_object()) {
    throw std::invalid_argument("Expected line to be a json object.");
  }

  auto [tokens, predict_start] = getContext(line_content);

  BoltVector prompt = promptContext(getPrompt(line_content));

  std::vector<std::vector<BoltVector>> vectors;

  for (uint32_t i = predict_start; i < tokens.size(); i++) {
    BoltVector label = BoltVector::singleElementSparseVector(tokens[i]);

    vectors.push_back({prompt, _context_featurizer.lrcContext(tokens, 0, i),
                       _context_featurizer.ircContext(tokens, 0, i),
                       _context_featurizer.srcContext(tokens, 0, i),
                       std::move(label)});
  }

  return vectors;
}

BoltVector ContextAwareTextFeaturizer::promptContext(
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

std::vector<BoltVector> ContextAwareTextFeaturizer::featurizeInferenceSample(
    const std::vector<uint32_t>& prompt,
    const std::vector<uint32_t>& context) const {
  return {promptContext(prompt), _context_featurizer.lrcContext(context),
          _context_featurizer.ircContext(context),
          _context_featurizer.srcContext(context)};
}

std::pair<std::vector<uint32_t>, uint32_t>
ContextAwareTextFeaturizer::getContext(const json& line_content) {
  if (!line_content.contains("target")) {
    throw std::invalid_argument("Expected field 'target' in json object'");
  }

  std::vector<uint32_t> target_tokens =
      token_encoding::tokenIds(thirdai::dataset::TextGenerationFeaturizer::getStringField(line_content, "target"));

  if (line_content.contains("context")) {
    std::vector<uint32_t> context_tokens =
        token_encoding::tokenIds(thirdai::dataset::TextGenerationFeaturizer::getStringField(line_content, "context"));

    // This assumes the we dont have any token to skip at the start of target.
    uint32_t predict_start = context_tokens.size();

    context_tokens.insert(context_tokens.end(), target_tokens.begin(),
                          target_tokens.end());

    return {std::move(context_tokens), predict_start};
  }

  return {std::move(target_tokens), 1};
}

std::vector<uint32_t> ContextAwareTextFeaturizer::getPrompt(
    const json& line_content) {
  if (line_content.contains("prompt")) {
    return token_encoding::tokenIds(thirdai::dataset::TextGenerationFeaturizer::getStringField(line_content, "prompt"));
  }
  return {};
}

void ContextAwareTextFeaturizer::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void ContextAwareTextFeaturizer::save_stream(
    std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

ContextAwareTextFeaturizerPtr ContextAwareTextFeaturizer::load(
    const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

ContextAwareTextFeaturizerPtr ContextAwareTextFeaturizer::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<ContextAwareTextFeaturizer> deserialize_into(
      new ContextAwareTextFeaturizer());
  iarchive(*deserialize_into);
  return deserialize_into;
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::ContextAwareTextFeaturizer)
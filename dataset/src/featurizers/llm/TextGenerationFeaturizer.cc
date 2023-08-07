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
  if (_context_featurizer.needPositionContext()) {
    data.resize(6);
  }

  for (auto& vectors : featurized_samples) {
    for (auto& sample : vectors) {
      for (uint32_t i = 0; i < sample.size(); i++) {
        data.at(i).push_back(std::move(sample[i]));
      }
    }
  }

  return data;
}

std::string getStringField(const json& json_object, const std::string& name) {
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

  auto [tokens, predict_start] = getContext(line_content);

  BoltVector prompt = promptContext(getPrompt(line_content));

  std::vector<std::vector<BoltVector>> vectors;

  uint32_t context_length = _context_featurizer.getLongRangeContextLength();

  for (uint32_t i = predict_start; i < tokens.size(); i += context_length) {
    uint32_t end_index =
        std::min((i + context_length), static_cast<uint32_t>(tokens.size()));

    for (uint32_t j = i; j < end_index; j++) {
      BoltVector label = BoltVector::singleElementSparseVector(tokens[j]);

      // start index = i-1, since we maintain an offset of 1 with predict_start
      // and i-1 is always >= 0
      std::vector<BoltVector> featurized_vectors = {
          prompt,
          _context_featurizer.lrcContext(tokens, j, i - 1),
          _context_featurizer.ircContext(tokens, j, i - 1),
          _context_featurizer.srcContext(tokens, j, i - 1),
      };

      if (_context_featurizer.needPositionContext()) {
        featurized_vectors.push_back(
            _context_featurizer.positionContext(tokens, j, i - 1));
      }

      featurized_vectors.push_back(std::move(label));
      vectors.push_back(featurized_vectors);
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
  std::vector<BoltVector> inference_sample = {
      promptContext(prompt), _context_featurizer.lrcContext(context),
      _context_featurizer.ircContext(context),
      _context_featurizer.srcContext(context)};
  if (_context_featurizer.needPositionContext()) {
    inference_sample.push_back(_context_featurizer.positionContext(context));
  }
  return inference_sample;
}

std::pair<std::vector<uint32_t>, uint32_t> TextGenerationFeaturizer::getContext(
    const json& line_content) {
  if (!line_content.contains("target")) {
    throw std::invalid_argument("Expected field 'target' in json object'");
  }

  std::vector<uint32_t> target_tokens =
      token_encoding::tokenIds(getStringField(line_content, "target"));

  if (line_content.contains("context")) {
    std::vector<uint32_t> context_tokens =
        token_encoding::tokenIds(getStringField(line_content, "context"));

    // The predict start is 1 after the end of the context because there will be
    // a [CLS] token.
    uint32_t predict_start = context_tokens.size() + 1;

    context_tokens.insert(context_tokens.end(), target_tokens.begin(),
                          target_tokens.end());

    return {std::move(context_tokens), predict_start};
  }

  return {std::move(target_tokens), 1};
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
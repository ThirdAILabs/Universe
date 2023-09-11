#include "TextGenerationFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <bolt/src/train/trainer/Dataset.h>
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
    if (_featurize_in_chunks) {
      featurized_samples[i] = featurizeTextChunks(lines[i]);
    } else {
      featurized_samples[i] = featurizeTextSlidingWindow(lines[i]);
    }
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

std::string getStringField(const json& json_object, const std::string& name) {
  if (!json_object[name].is_string()) {
    throw std::invalid_argument("Expected field '" + name +
                                "' to be a string.");
  }
  return json_object[name].get<std::string>();
}

std::vector<std::vector<BoltVector>>
TextGenerationFeaturizer::featurizeTextChunks(const std::string& line) const {
  auto line_content = json::parse(line);
  if (!line_content.is_object()) {
    throw std::invalid_argument("Expected line to be a json object.");
  }

  auto [tokens, _] = getAllTokens(line_content, /* with_context= */ false);

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

std::vector<std::vector<BoltVector>>
TextGenerationFeaturizer::featurizeTextSlidingWindow(
    const std::string& line) const {
  auto line_content = json::parse(line);
  if (!line_content.is_object()) {
    throw std::invalid_argument("Expected line to be a json object.");
  }

  auto [tokens, context_size] =
      getAllTokens(line_content, /* with_context= */ true);

  BoltVector prompt = promptContext(getPrompt(line_content));

  std::vector<std::vector<BoltVector>> vectors;

  for (size_t i = std::max<size_t>(context_size, 1); i < tokens.size(); i++) {
    BoltVector label = BoltVector::singleElementSparseVector(tokens[i]);

    vectors.push_back({prompt, _context_featurizer.lrcContext(tokens, 0, i),
                       _context_featurizer.ircContext(tokens, 0, i),
                       _context_featurizer.srcContext(tokens, 0, i),
                       std::move(label)});
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

bolt::TensorList TextGenerationFeaturizer::featurizeInputBatch(
    const std::vector<std::vector<uint32_t>>& tokens,
    const std::vector<uint32_t>& dims) const {
  std::vector<BoltVector> lrc;
  lrc.reserve(tokens.size());
  std::vector<BoltVector> irc;
  irc.reserve(tokens.size());
  std::vector<BoltVector> src;
  src.reserve(tokens.size());

  for (const auto& sample : tokens) {
    lrc.emplace_back(_context_featurizer.lrcContext(sample));
    irc.emplace_back(_context_featurizer.ircContext(sample));
    src.emplace_back(_context_featurizer.srcContext(sample));
  }

  return bolt::convertBatch(
      {BoltBatch(std::move(lrc)), BoltBatch(std::move(irc)),
       BoltBatch(std::move(src))},
      dims);
}

std::pair<std::vector<uint32_t>, size_t> TextGenerationFeaturizer::getAllTokens(
    const json& line_content, bool with_context) {
  if (!line_content.contains("target")) {
    throw std::invalid_argument("Expected field 'target' in json object'");
  }

  std::vector<uint32_t> target_tokens =
      token_encoding::tokenIds(getStringField(line_content, "target"));

  if (line_content.contains("context") && with_context) {
    std::vector<uint32_t> context_tokens =
        token_encoding::tokenIds(getStringField(line_content, "context"));

    size_t context_size = context_tokens.size();

    context_tokens.insert(context_tokens.end(), target_tokens.begin(),
                          target_tokens.end());

    return {std::move(context_tokens), context_size};
  }

  return {std::move(target_tokens), 0};
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
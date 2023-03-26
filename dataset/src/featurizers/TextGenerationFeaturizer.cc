#include "TextGenerationFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/HashUtils.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <json/include/nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

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

  if (!line_content.contains("target")) {
    throw std::invalid_argument("Expected field 'target' in json object'");
  }
  auto target = getStringField(line_content, "target");

  std::vector<uint32_t> target_tokens = parseTokens(target);

  std::vector<uint32_t> tokens;
  uint32_t predict_start;
  if (line_content.contains("context")) {
    tokens = parseTokens(getStringField(line_content, "context"));
    // The predict start is 1 after the end of the context because there will be
    // a [CLS] token.
    predict_start = tokens.size() + 1;
    tokens.insert(tokens.end(), target_tokens.begin(), target_tokens.end());
  } else {
    tokens = std::move(target_tokens);
    // The predict start is 1 instead of 0 because there will be a [CLS] token.
    predict_start = 1;
  }

  BoltVector prompt;
  if (line_content.contains("prompt")) {
    std::vector<uint32_t> prompt_tokens =
        parseTokens(getStringField(line_content, "prompt"));

    prompt = BoltVector(/* l= */ prompt_tokens.size(), /* is_dense= */ false,
                        /* has_gradient= */ false);
    std::copy(prompt_tokens.begin(), prompt_tokens.end(),
              prompt.active_neurons);
    std::fill_n(prompt.activations, prompt.len, 1.0);
  } else {
    prompt = BoltVector::singleElementSparseVector(0);
  }

  std::vector<std::vector<BoltVector>> vectors;

  for (uint32_t i = predict_start; i < tokens.size(); i++) {
    BoltVector label = BoltVector::singleElementSparseVector(tokens[i]);

    vectors.push_back({prompt, lrcContext(tokens, i), ircContext(tokens, i),
                       srcContext(tokens, i), std::move(label)});
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

BoltVector TextGenerationFeaturizer::lrcContext(
    const std::vector<uint32_t>& tokens, uint32_t label_index) const {
  uint32_t lrc_len = std::min(label_index, _lrc_len);

  const uint32_t* context_start = tokens.data() + label_index - lrc_len;

  BoltVector vector(/* l= */ lrc_len, /* is_dense= */ false,
                    /* has_gradient= */ false);
  std::copy(context_start, context_start + lrc_len, vector.active_neurons);
  std::fill_n(vector.activations, vector.len, 1.0);

  return vector;
}

BoltVector TextGenerationFeaturizer::ircContext(
    const std::vector<uint32_t>& tokens, uint32_t label_index) const {
  uint32_t irc_len = std::min(label_index, _irc_len);

  std::vector<uint32_t> irc_context =
      token_encoding::unigramPreservingPairgrams(
          tokens.data() + label_index - irc_len, irc_len, _vocab_size);

  BoltVector vector(/* l= */ irc_context.size(), /* is_dense= */ false,
                    /* has_gradient= */ false);
  std::copy(irc_context.begin(), irc_context.end(), vector.active_neurons);
  std::fill_n(vector.activations, vector.len, 1.0);

  return vector;
}

BoltVector TextGenerationFeaturizer::srcContext(
    const std::vector<uint32_t>& tokens, uint32_t label_index) const {
  uint32_t src_len = std::min(label_index, _src_len);
  uint32_t padding_len = _src_len - src_len;

  const uint32_t* context_start = tokens.data() + label_index - src_len;

  BoltVector vector(/* l= */ _src_len, /* is_dense= */ false,
                    /* has_gradient= */ false);

  // Zero pad if short range context length is greater than number of tokens. We
  // pad the begining so that the last token before the prediction is always at
  // the end.
  std::fill_n(vector.active_neurons, padding_len, 0);
  std::copy(context_start, context_start + src_len,
            vector.active_neurons + padding_len);
  std::fill_n(vector.activations, vector.len, 1.0);

  return vector;
}

std::vector<BoltVector> TextGenerationFeaturizer::featurizeInferenceSample(
    const std::vector<uint32_t>& tokens) const {
  uint32_t prediction_index = tokens.size();
  return {lrcContext(tokens, prediction_index),
          ircContext(tokens, prediction_index),
          srcContext(tokens, prediction_index)};
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
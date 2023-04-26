#include "DataTypes.h"
#include <cereal/archives/binary.hpp>

namespace thirdai::automl::data {

dataset::TextTokenizerPtr getTextTokenizerFromString(
    const std::string& string) {
  if (std::regex_match(string, std::regex("char-[1-9]\\d*"))) {
    uint32_t k = std::strtol(string.data() + 5, nullptr, 10);
    return dataset::CharKGramTokenizer::make(/* k = */ k);
  }

  if (string == "words") {
    return dataset::NaiveSplitTokenizer::make();
  }

  if (string == "words-punct") {
    return dataset::WordPunctTokenizer::make();
  }

  throw std::invalid_argument(
      "Created text column with invalid tokenizer '" + string +
      "', please choose one of 'words', 'words-punct', or 'char-k' "
      "(k is a number, e.g. 'char-5').");
}

dataset::TextEncoderPtr getTextEncoderFromString(const std::string& string) {
  std::unordered_map<std::string, dataset::TextEncoderPtr>
      contextual_encodings = {
          {"none", dataset::NGramEncoder::make(/* n = */ 1)},
          {"local", dataset::NGramEncoder::make(/* n = */ 2)},
          {"global", dataset::PairGramEncoder::make()},
      };

  if (contextual_encodings.count(string) == 0) {
    throw std::invalid_argument(
        "Created text column with invalid contextual_encoding '" + string +
        "', please choose one of 'none', 'local', or 'global'.");
  };

  return contextual_encodings[string];
}

CategoricalDataTypePtr asCategorical(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<CategoricalDataType>(data_type);
}

NumericalDataTypePtr asNumerical(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<NumericalDataType>(data_type);
}

TextDataTypePtr asText(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<TextDataType>(data_type);
}

DateDataTypePtr asDate(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<DateDataType>(data_type);
}

SequenceDataTypePtr asSequence(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<SequenceDataType>(data_type);
}

NeighborsDataTypePtr asNeighbors(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<NeighborsDataType>(data_type);
}

NodeIDDataTypePtr asNodeID(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<NodeIDDataType>(data_type);
}

}  // namespace thirdai::automl::data

CEREAL_REGISTER_TYPE(thirdai::automl::data::CategoricalDataType)
CEREAL_REGISTER_TYPE(thirdai::automl::data::NumericalDataType)
CEREAL_REGISTER_TYPE(thirdai::automl::data::DateDataType)
CEREAL_REGISTER_TYPE(thirdai::automl::data::TextDataType)
CEREAL_REGISTER_TYPE(thirdai::automl::data::SequenceDataType)
CEREAL_REGISTER_TYPE(thirdai::automl::data::NeighborsDataType)
CEREAL_REGISTER_TYPE(thirdai::automl::data::NodeIDDataType)

#include "DataTypes.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <utils/Random.h>

namespace thirdai::automl {

dataset::TextTokenizerPtr getTextTokenizerFromString(const std::string& string,
                                                     uint32_t seed) {
  if (std::regex_match(string, std::regex("char-[1-9]\\d*"))) {
    uint32_t k = std::strtol(string.data() + 5, nullptr, 10);
    return dataset::CharKGramTokenizer::make(
        /* k = */ k, /*seed=*/seed);
  }

  if (string == "words") {
    return dataset::NaiveSplitTokenizer::make(
        /*delimiter=*/' ', /*seed=*/seed);
  }

  if (string == "words-punct") {
    return dataset::WordPunctTokenizer::make(
        /*seed=*/seed);
  }

  throw std::invalid_argument(
      "Created text column with invalid tokenizer '" + string +
      "', please choose one of 'words', 'words-punct', or 'char-k' "
      "(k is a number, e.g. 'char-5').");
}

dataset::TextEncoderPtr getTextEncoderFromString(const std::string& string) {
  if (std::regex_match(string, std::regex("ngram-(0|[1-9]\\d*)"))) {
    uint32_t n = std::strtol(string.data() + 6, nullptr, 10);
    if (n == 0) {
      throw std::invalid_argument(
          "Specified 'ngram-N' option with N = 0. Please use N > 0.");
    }
    return dataset::NGramEncoder::make(/* n = */ n);
  }

  std::unordered_map<std::string, dataset::TextEncoderPtr>
      contextual_encodings = {
          {"none", dataset::NGramEncoder::make(/* n = */ 1)},
          {"local", dataset::NGramEncoder::make(/* n = */ 2)},
          {"global", dataset::PairGramEncoder::make()},
      };

  if (contextual_encodings.count(string) == 0) {
    throw std::invalid_argument(
        "Created text column with invalid contextual_encoding '" + string +
        "', please choose one of 'none', 'local', 'ngram-N', or 'global'.");
  };

  return contextual_encodings[string];
}

uint32_t NumericalDataType::numBins() const {
  auto lower_size = text::lower(granularity);
  if (lower_size == "xs" || lower_size == "extrasmall") {
    return 10;
  }
  if (lower_size == "s" || lower_size == "small") {
    return 75;
  }
  if (lower_size == "m" || lower_size == "medium") {
    return 300;
  }
  if (lower_size == "l" || lower_size == "large") {
    return 1000;
  }
  if (lower_size == "xl" || lower_size == "extralarge") {
    return 3000;
  }
  throw std::invalid_argument("Invalid numerical granularity \"" + granularity +
                              "\". Choose one of \"extrasmall\"/\"xs\", "
                              "\"small\"/\"s\", \"medium\"/\"m\", "
                              "\"large\"/\"l\", or \"extralarge\"/\"xl\".");
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

}  // namespace thirdai::automl

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::automl::CategoricalDataType,
                               "thirdai::automl::data::CategoricalDataType")
CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::automl::NumericalDataType,
                               "thirdai::automl::data::NumericalDataType")
CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::automl::DateDataType,
                               "thirdai::automl::data::DateDataType")
CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::automl::TextDataType,
                               "thirdai::automl::data::TextDataType")
CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::automl::SequenceDataType,
                               "thirdai::automl::data::SequenceDataType")
CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::automl::NeighborsDataType,
                               "thirdai::automl::data::NeighborsDataType")
CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::automl::NodeIDDataType,
                               "thirdai::automl::data::NodeIDDataType")

#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <utils/Logging.h>
#include <utils/StringManipulation.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace thirdai::automl::data {

struct CategoricalMetadataConfig;
using CategoricalMetadataConfigPtr = std::shared_ptr<CategoricalMetadataConfig>;

struct DataType {
  virtual std::string toString() const = 0;

  virtual ~DataType() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using DataTypePtr = std::shared_ptr<DataType>;

struct CategoricalDataType : DataType {
  explicit CategoricalDataType(std::optional<char> delimiter = std::nullopt,
                               CategoricalMetadataConfigPtr metadata = nullptr)
      : delimiter(delimiter), metadata_config(std::move(metadata)) {}

  std::optional<char> delimiter;
  CategoricalMetadataConfigPtr metadata_config;

  std::string toString() const final {
    if (delimiter.has_value()) {
      return fmt::format(R"({{"type": "categorical", "delimiter": "{}"}})",
                         delimiter.value());
    }
    return fmt::format(R"({{"type": "categorical"}})");
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this), delimiter, metadata_config);
  }
};

using CategoricalDataTypePtr = std::shared_ptr<CategoricalDataType>;

enum class TextEncodingType {
  Unigrams,
  Bigrams,
  Pairgrams,
};

inline TextEncodingType getTextEncodingFromString(const std::string& encoding) {
  std::unordered_map<std::string, TextEncodingType> contextual_encodings = {
      {"none", TextEncodingType::Unigrams},
      {"local", TextEncodingType::Bigrams},
      {"global", TextEncodingType::Pairgrams},
  };

  if (contextual_encodings.count(encoding) == 0) {
    throw std::invalid_argument(
        "Created text column with invalid contextual_encoding '" + encoding +
        "' please choose one of 'none', 'local', or 'global'.");
  };
  return contextual_encodings[encoding];
}

struct TextDataType : DataType {
  explicit TextDataType(std::optional<double> average_n_words = std::nullopt,
                        const std::string& contextual_encoding = "none")
      : average_n_words(average_n_words),
        contextual_encoding(getTextEncodingFromString(contextual_encoding)) {}

  std::optional<double> average_n_words;
  TextEncodingType contextual_encoding;

  std::string toString() const final { return R"({"type": "text"})"; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this), average_n_words,
            contextual_encoding);
  }
};

using TextDataTypePtr = std::shared_ptr<TextDataType>;

struct NumericalDataType : DataType {
  explicit NumericalDataType(std::pair<double, double> _range,
                             std::string _granularity = "m")
      : range(std::move(_range)), granularity(std::move(_granularity)) {}

  std::pair<double, double> range;
  std::string granularity;

  NumericalDataType() {}

  std::string toString() const final {
    return fmt::format(
        R"({{"type": "numerical", "range": [{}, {}], "granularity": "{}"}})",
        range.first, range.second, granularity);
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this), range, granularity);
  }
};

using NumericalDataTypePtr = std::shared_ptr<NumericalDataType>;

struct DateDataType : DataType {
  std::string toString() const final { return R"({"type": "date"})"; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this));
  }
};

using DateDataTypePtr = std::shared_ptr<DateDataType>;

struct NeighborsDataType : DataType {
  std::string toString() const final { return R"({"type": "neighbors"})"; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this));
  }
};

using NeighborsDataTypePtr = std::shared_ptr<NeighborsDataType>;

struct NodeIDDataType : DataType {
  std::string toString() const final { return R"({"type": "node id"})"; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this));
  }
};

using NodeIDDataTypePtr = std::shared_ptr<NodeIDDataType>;

CategoricalDataTypePtr asCategorical(const DataTypePtr& data_type);

NumericalDataTypePtr asNumerical(const DataTypePtr& data_type);

TextDataTypePtr asText(const DataTypePtr& data_type);

DateDataTypePtr asDate(const DataTypePtr& data_type);

NeighborsDataTypePtr asNeighbors(const DataTypePtr& data_type);

NodeIDDataTypePtr asNodeID(const DataTypePtr& data_type);

using ColumnDataTypes = std::map<std::string, DataTypePtr>;

struct CategoricalMetadataConfig {
  std::string metadata_file;
  std::string key;
  ColumnDataTypes column_data_types;
  char delimiter = ',';

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(metadata_file, key, column_data_types, delimiter);
  }
};

struct TemporalCategoricalConfig {
  const std::string& column_name;
  uint32_t track_last_n;
  bool include_current_row;
  bool use_metadata;
};

struct TemporalNumericalConfig {
  const std::string& column_name;
  uint32_t history_length;
  bool include_current_row;
};

enum class TemporalType { categorical, numerical, no_type };

// TODO(Geordie): Instead of having this all-knowing method class
// we should just use interface + dynamic casting.
class TemporalConfig {
 public:
  TemporalConfig() : _type(TemporalType::no_type) {}

  static auto categorical(std::string column_name, uint32_t track_last_n,
                          bool include_current_row = false,
                          bool use_metadata = false) {
    return TemporalConfig(TemporalType::categorical, std::move(column_name),
                          /* track_last_n= */ track_last_n,
                          /* history_length= */ 0,
                          /* include_current_row= */ include_current_row,
                          /* use_metadata= */ use_metadata);
  }

  static auto numerical(std::string column_name, uint32_t history_length,
                        bool include_current_row = false) {
    return TemporalConfig(TemporalType::numerical, std::move(column_name),
                          /* track_last_n= */ 0,
                          /* history_length= */ history_length,
                          /* include_current_row= */ include_current_row);
  }

  const std::string& columnName() const { return _column_name; }

  bool includesCurrentRow() const { return _include_current_row; }

  bool isCategorical() const { return _type == TemporalType::categorical; }
  bool isNumerical() const { return _type == TemporalType::numerical; }

  TemporalCategoricalConfig asCategorical() const {
    if (!isCategorical()) {
      throw std::invalid_argument(
          "[TemporalConfig] Tried to cast non-categorical config as a "
          "categorical config.");
    }
    return {_column_name, _track_last_n, _include_current_row, _use_metadata};
  }

  TemporalNumericalConfig asNumerical() const {
    if (!isNumerical()) {
      throw std::invalid_argument(
          "[TemporalConfig] Tried to cast non-numerical config as a numerical "
          "config.");
    }
    return {_column_name, _history_length, _include_current_row};
  }

 private:
  TemporalConfig(TemporalType type, std::string column_name,
                 uint32_t track_last_n, uint32_t history_length,
                 bool include_current_row, bool use_metadata = false)
      : _type(type),
        _column_name(std::move(column_name)),
        _track_last_n(track_last_n),
        _history_length(history_length),
        _include_current_row(include_current_row),
        _use_metadata(use_metadata) {}

  TemporalType _type;
  std::string _column_name;
  uint32_t _track_last_n;
  uint32_t _history_length;
  bool _include_current_row;
  bool _use_metadata;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_type, _column_name, _track_last_n, _history_length,
            _include_current_row);
  }
};

using UserProvidedTemporalRelationships =
    std::map<std::string,
             std::vector<std::variant<std::string, TemporalConfig>>>;

using TemporalRelationships =
    std::map<std::string, std::vector<TemporalConfig>>;

}  // namespace thirdai::automl::data

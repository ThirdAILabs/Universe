#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
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

namespace thirdai::automl::deployment {

struct CategoricalMetadataConfig;

using CategoricalMetadataConfigPtr = std::shared_ptr<CategoricalMetadataConfig>;

struct CategoricalDataType {
  explicit CategoricalDataType(std::optional<uint32_t> n_unique_classes,
                               std::optional<char> delimiter,
                               CategoricalMetadataConfigPtr metadata,
                               bool contiguous_numerical_ids)
      : n_unique_classes(n_unique_classes),
        delimiter(delimiter),
        metadata_config(std::move(metadata)),
        contiguous_numerical_ids(contiguous_numerical_ids) {}

  std::optional<uint32_t> n_unique_classes;
  std::optional<char> delimiter;
  CategoricalMetadataConfigPtr metadata_config;
  bool contiguous_numerical_ids;

  CategoricalDataType() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(n_unique_classes, delimiter, metadata_config,
            contiguous_numerical_ids);
  }
};

struct TextDataType {
  explicit TextDataType(std::optional<uint32_t> average_n_words,
                        const std::string& embedding_size, bool force_pairgram)
      : average_n_words(average_n_words), force_pairgram(force_pairgram) {
    auto embedding_size_lower = utils::lower(embedding_size);
    if (embedding_size_lower == "s" || embedding_size_lower == "small") {
      this->dim = 30000;
    } else if (embedding_size_lower == "m" ||
               embedding_size_lower == "medium") {
      this->dim = 100000;
    } else if (embedding_size_lower == "l" || embedding_size_lower == "large") {
      this->dim = 500000;
    } else {
      throw std::invalid_argument(
          embedding_size +
          " is not a valid embedding size option. Choose between 'small'/'s', "
          "'medium'/'m', and 'large'/'l'.");
    }
  }
  std::optional<uint32_t> average_n_words;
  uint32_t dim;
  bool force_pairgram;

  TextDataType() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(average_n_words, dim, force_pairgram);
  }
};

struct NumericalDataType {
  explicit NumericalDataType(std::pair<double, double> _range,
                             std::string _granularity)
      : range(std::move(_range)), granularity(std::move(_granularity)) {}

  std::pair<double, double> range;
  std::string granularity;

  NumericalDataType() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(range, granularity);
  }
};

struct DateDataType {};

struct NoneDataType {};

using AnyDataType = std::variant<NoneDataType, DateDataType, NumericalDataType,
                                 CategoricalDataType, TextDataType>;

// TODO(Geordie): Instead of having this all-knowing method class
// we should just use interface + dynamic casting.
class DataType {
 public:
  DataType() : _value(NoneDataType()) {}

  static auto categorical(
      std::optional<uint32_t> n_unique_classes = std::nullopt,
      std::optional<char> delimiter = std::nullopt,
      CategoricalMetadataConfigPtr metadata = nullptr,
      bool contiguous_numerical_ids = false) {
    return DataType(CategoricalDataType(n_unique_classes, delimiter,
                                        std::move(metadata),
                                        contiguous_numerical_ids));
  }

  static auto text(std::optional<uint32_t> average_n_words = std::nullopt,
                   const std::string& embedding_size = "m",
                   bool use_attention = false) {
    return DataType(TextDataType(average_n_words, embedding_size,
                                 /* force_pairgram= */ use_attention));
  }

  static auto numerical(std::pair<double, double> range,
                        std::string granularity = "m") {
    return DataType(NumericalDataType(range, std::move(granularity)));
  }

  static auto date() { return DataType(DateDataType()); }

  bool isCategorical() const {
    return std::holds_alternative<CategoricalDataType>(_value);
  }
  bool isNumerical() const {
    return std::holds_alternative<NumericalDataType>(_value);
  }
  bool isText() const { return std::holds_alternative<TextDataType>(_value); }
  bool isDate() const { return std::holds_alternative<DateDataType>(_value); }

  const CategoricalDataType& asCategorical() const {
    if (!isCategorical()) {
      throwCastError("categorical");
    }
    return std::get<CategoricalDataType>(_value);
  }

  const TextDataType& asText() const {
    if (!isText()) {
      throwCastError("text");
    }
    return std::get<TextDataType>(_value);
  }

  const NumericalDataType& asNumerical() const {
    if (!isNumerical()) {
      throwCastError("numerical");
    }
    return std::get<NumericalDataType>(_value);
  }

  const DateDataType& asDate() const {
    if (!isDate()) {
      throwCastError("date");
    }
    return std::get<DateDataType>(_value);
  }

 private:
  static void throwCastError(std::string&& type_name) {
    throw std::invalid_argument("[DataType] Tried to cast non-" + type_name +
                                " datatype as a " + type_name + " datatype.");
  }

  explicit DataType(AnyDataType value) : _value(std::move(value)) {}

  AnyDataType _value;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_value);
  }
};

using ColumnDataTypes = std::map<std::string, DataType>;

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

}  // namespace thirdai::automl::deployment
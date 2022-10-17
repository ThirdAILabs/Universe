#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/variant.hpp>
#include <utils/StringManipulation.h>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace thirdai::bolt::sequential_classifier {

struct CategoricalDataType {
  explicit CategoricalDataType(uint32_t n_unique_classes,
                               std::optional<char> delimiter)
      : n_unique_classes(n_unique_classes), delimiter(delimiter) {}

  uint32_t n_unique_classes;
  std::optional<char> delimiter;

 private:
  // Private constructor for Cereal.
  CategoricalDataType() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(n_unique_classes, delimiter);
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

 private:
  // Private constructor for Cereal
  TextDataType() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(average_n_words, dim, force_pairgram);
  }
};

struct NumericalDataType {};

struct DateDataType {};

struct NoneDataType {};

using AnyDataType = std::variant<CategoricalDataType, TextDataType,
                                 NumericalDataType, DateDataType, NoneDataType>;

class DataType {
 public:
  DataType() : _value(NoneDataType()) {}

  static auto categorical(uint32_t n_unique_classes,
                          std::optional<char> delimiter = std::nullopt) {
    return DataType(CategoricalDataType(n_unique_classes, delimiter));
  }

  static auto text(std::optional<uint32_t> average_n_words = std::nullopt,
                   const std::string& embedding_size = "m",
                   bool use_attention = false) {
    return DataType(TextDataType(average_n_words, embedding_size,
                                 /* force_pairgram= */ use_attention));
  }

  static auto numerical() { return DataType(NumericalDataType()); }

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

  explicit DataType(AnyDataType value) : _value(value) {}

  AnyDataType _value;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_value);
  }
};

struct TemporalCategoricalConfig {
  const std::string& column_name;
  uint32_t track_last_n;
  bool include_current_row;
};

struct TemporalNumericalConfig {
  const std::string& column_name;
  uint32_t history_length;
  bool include_current_row;
};

enum class TemporalType { categorical, numerical, no_type };

class TemporalConfig {
 public:
  TemporalConfig() : _type(TemporalType::no_type) {}

  static auto categorical(std::string column_name, uint32_t track_last_n,
                          bool include_current_row = false) {
    return TemporalConfig(TemporalType::categorical, std::move(column_name),
                          /* track_last_n= */ track_last_n,
                          /* history_length= */ 0,
                          /* include_current_row= */ include_current_row);
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
    return {_column_name, _track_last_n, _include_current_row};
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
                 bool include_current_row)
      : _type(type),
        _column_name(std::move(column_name)),
        _track_last_n(track_last_n),
        _history_length(history_length),
        _include_current_row(include_current_row) {}

  TemporalType _type;
  std::string _column_name;
  uint32_t _track_last_n;
  uint32_t _history_length;
  bool _include_current_row;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_type, _column_name, _track_last_n, _history_length,
            _include_current_row);
  }
};

}  // namespace thirdai::bolt::sequential_classifier
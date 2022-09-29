#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::bolt::sequential_classifier {

struct CategoricalDataType {
  uint32_t n_unique_classes;
};

struct TextDataType {
  std::optional<uint32_t> average_n_words;
};

struct NumericalDataType {};

struct DateDataType {};

enum class Type { categorical, text, numerical, date, no_type };

class DataType {
 public:
  DataType() : _type(Type::no_type) {}

  static auto categorical(uint32_t n_unique_classes) {
    return DataType(/* type= */ Type::categorical,
                    /* n_unique_classes= */ n_unique_classes,
                    /* average_n_words= */ 0);
  }

  static auto text(std::optional<uint32_t> average_n_words = std::nullopt) {
    return DataType(/* type= */ Type::text, /* n_unique_classes= */ 0,
                    /* average_n_words= */ average_n_words);
  }

  static auto numerical() {
    return DataType(/* type= */ Type::numerical, /* n_unique_classes= */ 0,
                    /* average_n_words= */ 0);
  }

  static auto date() {
    return DataType(/* type= */ Type::date, /* n_unique_classes= */ 0,
                    /* average_n_words= */ 0);
  }

  bool isCategorical() const { return _type == Type::categorical; }
  bool isNumerical() const { return _type == Type::numerical; }
  bool isText() const { return _type == Type::text; }
  bool isDate() const { return _type == Type::date; }

  CategoricalDataType asCategorical() const {
    if (!isCategorical()) {
      throw std::invalid_argument(
          "[DataType] Tried to cast non-categorical datatype as a categorical "
          "datatype.");
    }
    return {_n_unique_classes};
  }

  TextDataType asText() const {
    if (!isText()) {
      throw std::invalid_argument(
          "[DataType] Tried to cast non-text datatype as a text datatype.");
    }
    return {_average_n_words};
  }

  NumericalDataType asNumerical() const {
    if (!isNumerical()) {
      throw std::invalid_argument(
          "[DataType] Tried to cast non-numerical datatype as a numerical "
          "datatype.");
    }
    return {};
  }

  DateDataType asDate() const {
    if (!isDate()) {
      throw std::invalid_argument(
          "[DataType] Tried to cast non-date datatype as a date datatype.");
    }
    return {};
  }

 private:
  explicit DataType(Type type, uint32_t n_unique_classes,
                    std::optional<uint32_t> average_n_words)
      : _type(type),
        _n_unique_classes(n_unique_classes),
        _average_n_words(average_n_words) {}

  Type _type;
  uint32_t _n_unique_classes;
  std::optional<uint32_t> _average_n_words;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_type, _n_unique_classes, _average_n_words);
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
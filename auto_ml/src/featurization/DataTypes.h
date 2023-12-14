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
#include <dataset/src/blocks/text/HybridTokenizer.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>
#include <utils/Logging.h>
#include <utils/StringManipulation.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace thirdai::automl {

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

struct CategoricalDataType final : public DataType {
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

dataset::TextTokenizerPtr getTextTokenizerFromString(const std::string& string);

dataset::TextEncoderPtr getTextEncoderFromString(const std::string& string);

struct TextDataType final : public DataType {
  explicit TextDataType(const std::string& tokenizer = "words",
                        const std::string& contextual_encoding = "none",
                        bool use_lowercase = true)
      : tokenizer(getTextTokenizerFromString(tokenizer)),
        encoder(getTextEncoderFromString(contextual_encoding)),
        lowercase(use_lowercase) {}

  explicit TextDataType(
      const dataset::WordpieceTokenizerPtr& wordpiece_tokenizer,
      const std::string& contextual_encoding = "none")
      : tokenizer(wordpiece_tokenizer),
        encoder(getTextEncoderFromString(contextual_encoding)),
        // in this case the wordpiece tokenizer handles the lowercasing
        lowercase(false) {}

  explicit TextDataType(
      const dataset::HybridWordpieceCharKTokenizerPtr& hybrid_tokenizer,
      const std::string& contextual_encoding = "none")
      : tokenizer(hybrid_tokenizer),
        encoder(getTextEncoderFromString(contextual_encoding)),
        // in this case the hybrid tokenizer handles the lowercasing
        lowercase(false) {}

  dataset::TextTokenizerPtr tokenizer;
  dataset::TextEncoderPtr encoder;
  bool lowercase;

  std::string toString() const final { return R"({"type": "text"})"; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this), tokenizer, encoder, lowercase);
  }
};

using TextDataTypePtr = std::shared_ptr<TextDataType>;

struct NumericalDataType final : public DataType {
  explicit NumericalDataType(std::pair<double, double> _range,
                             std::string _granularity = "m")
      : range(std::move(_range)), granularity(std::move(_granularity)) {}

  NumericalDataType(double start, double end, std::string _granularity = "m")
      : range(start, end), granularity(std::move(_granularity)) {}

  std::pair<double, double> range;
  std::string granularity;

  NumericalDataType() {}

  std::string toString() const final {
    return fmt::format(
        R"({{"type": "numerical", "range": [{}, {}], "granularity": "{}"}})",
        range.first, range.second, granularity);
  }

  uint32_t numBins() const;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this), range, granularity);
  }
};

using NumericalDataTypePtr = std::shared_ptr<NumericalDataType>;

struct DateDataType final : public DataType {
  std::string toString() const final { return R"({"type": "date"})"; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this));
  }
};

using DateDataTypePtr = std::shared_ptr<DateDataType>;

struct SequenceDataType final : public DataType {
  explicit SequenceDataType(char delimiter = ' ',
                            std::optional<uint32_t> max_length = std::nullopt)
      : delimiter(delimiter), max_length(max_length) {
    if (max_length && max_length.value() == 0) {
      throw std::invalid_argument("Sequence max_length cannot be 0.");
    }
  }

  char delimiter;
  std::optional<uint32_t> max_length;

  std::string toString() const final { return R"({"type": "sequence"})"; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DataType>(this), delimiter, max_length);
  }
};

using SequenceDataTypePtr = std::shared_ptr<SequenceDataType>;

/**
 * Should only be used for graph data. Represents the neighbors of a node
 * as a space separated list of positive integers (each integer is the node id
 * of a neighbor; see NodeIDDataType). Thus, the first few lines of valid data
 * for a graph dataset might look like:
 *
 *        neighbors, node_id, target
 *        1 4 5,0,1
 *        ,1,0
 *        9 2,2,0
 * TODO(Any): Make the delimiter character configurable
 */
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

/**
 * Should only be used for graph data. Represents the id of a node
 * as a single integer. Each node (a row of input data) should have a unique
 * node id. Thus, the first few lines of valid data for a graph dataset might
 * look like:
 *
 *        neighbors, node_id, target
 *        1 4 5,0,1
 *        ,1,0
 *        9 2,2,0
 */
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

SequenceDataTypePtr asSequence(const DataTypePtr& data_type);

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

}  // namespace thirdai::automl

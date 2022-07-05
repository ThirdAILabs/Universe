#include "SchemaProcessor.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

// Declarations for helper functions.
static bool isValidKey(const std::string& key);
static void throwInvalidKeyError(const std::string& key);
static void throwMissingKeyError(const std::string& key);
static void throwMissingColumnError(const std::string& col_name);

static const std::unordered_map<std::string, SchemaKey>
    string_to_key = {
        {"user", SchemaKey::user},
        {"item", SchemaKey::item},
        {"timestamp", SchemaKey::timestamp},
        {"text_attr", SchemaKey::text_attr},
        {"categorical_attr", SchemaKey::categorical_attr},
        {"trackable_quantity", SchemaKey::trackable_quantity},
        {"target", SchemaKey::target}};

static const std::unordered_map<SchemaKey, std::string>
    required_keys_to_str = {
        {SchemaKey::item, "item"},
        {SchemaKey::timestamp, "timestamp"},
        {SchemaKey::target, "target"}};

SchemaProcessor::SchemaProcessor(GivenSchema& schema) {
  for (const auto& [key_str, name] : schema) {
    if (!isValidKey(key_str)) {
      throwInvalidKeyError(key_str);
    }
    _schema[string_to_key.at(key_str)] = name;
  }

  for (const auto& [key, key_str] : required_keys_to_str) {
    if (_schema.count(key) == 0) {
      throwMissingKeyError(key_str);
    }
  }
}

ColumnNumbers SchemaProcessor::parseHeader(const std::string& header,
                                           const char delimiter) {
  std::unordered_map<std::string, size_t> col_names_to_nums;
  size_t col = 0;
  size_t start = 0;
  size_t end = 0;
  while (end != std::string::npos) {
    end = header.find(delimiter, start);
    size_t len = end == std::string::npos ? header.size() - start : end - start;
    col_names_to_nums[header.substr(start, len)] = col;
    col++;
    start = end + 1;
  }

  ColumnNumbers keys_to_col_nums;
  for (const auto& [key, col_name] : _schema) {
    if (col_names_to_nums.count(col_name) == 0) {
      throwMissingColumnError(col_name);
    }
    keys_to_col_nums[key] = col_names_to_nums[col_name];
  }
  return keys_to_col_nums;
}

bool isValidKey(const std::string& key) {
  return string_to_key.count(key) > 0;
}

void throwInvalidKeyError(const std::string& key) {
  std::stringstream ss;
  ss << "Found invalid key '" << key << "' in schema. Valid keys: ";

  std::string delimiter;
  for (const auto& [key_str, _] : string_to_key) {
    ss << delimiter << "'" << key_str << "'";
    delimiter = ", ";
  }

  throw std::invalid_argument(ss.str());
}

void throwMissingKeyError(const std::string& key) {
  std::stringstream ss;
  ss << "Schema is missing required key '" << key
     << "' in schema. Required keys: ";

  std::string delimiter;
  for (const auto& [_, key_str] : required_keys_to_str) {
    ss << delimiter << "'" << key_str << "'";
    delimiter = ", ";
  }

  throw std::invalid_argument(ss.str());
}

void throwMissingColumnError(const std::string& col_name) {
  std::stringstream ss;
  ss << "Could not find a column named '" << col_name << "' in header.";
  throw std::invalid_argument(ss.str());
}

}  // namespace thirdai::bolt
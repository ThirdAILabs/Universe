#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

constexpr const char* key_re_str = "\\w+(?=\\s*=)";
constexpr const char* int_re_str = "\\d+(?=\\s*(,|$))";
constexpr const char* decimal_re_str = R"((\d+\.\d+)(?=\s*(,|$)))";
constexpr const char* string_re_str = R"(("|')[-\w\\/.]+("|')(?=\s*(,|$)))";
constexpr const char* comment_re_str = R"(^\s*\/\/.*)";
constexpr const char* empty_re_str = "^\\s*$";

class ConfigValue {
 public:
  virtual uint64_t intVal(uint32_t index) const {
    (void)index;
    throw std::logic_error(
        "Attempted to call IntVal on non integer config var.");
  }

  virtual double doubleVal(uint32_t index) const {
    (void)index;
    throw std::logic_error(
        "Attempted to call DoubleVal on non double config var.");
  }

  virtual const std::string& strVal(uint32_t index) const {
    (void)index;
    throw std::logic_error(
        "Attempted to call StrVal on non string config var.");
  }

  virtual std::ostream& print(std::ostream&) const = 0;

  friend std::ostream& operator<<(std::ostream& out, const ConfigValue& val);

  friend std::ostream& operator<<(std::ostream& out,
                                  const std::shared_ptr<ConfigValue>& val);

  virtual ~ConfigValue() {}
};

class IntValue final : public ConfigValue {
 public:
  explicit IntValue(std::vector<uint64_t>&& values) : values(values) {}

  uint64_t intVal(uint32_t index) const override { return values.at(index); }

  std::ostream& print(std::ostream& out) const override {
    for (const auto& val : values) {
      out << val << ", ";
    }
    return out;
  }

 private:
  std::vector<uint64_t> values;
};

class DoubleValue final : public ConfigValue {
 public:
  explicit DoubleValue(std::vector<double>&& values) : values(values) {}

  double doubleVal(uint32_t index) const override { return values.at(index); }

  std::ostream& print(std::ostream& out) const override {
    for (const auto& val : values) {
      out << val << ", ";
    }
    return out;
  }

 private:
  std::vector<double> values;
};

class StrValue final : public ConfigValue {
 public:
  explicit StrValue(std::vector<std::string>&& values) : values(values) {}

  const std::string& strVal(uint32_t index) const override {
    return values.at(index);
  }

  std::ostream& print(std::ostream& out) const override {
    for (const auto& val : values) {
      out << "'" << val << "', ";
    }
    return out;
  }

 private:
  std::vector<std::string> values;
};

class ConfigReader {
 public:
  explicit ConfigReader(const std::string& filename)
      : key_re(key_re_str),
        int_re(int_re_str),
        decimal_re(decimal_re_str),
        string_re(string_re_str),
        comment_re(comment_re_str),
        empty_re(empty_re_str) {
    parseConfig(filename);
  }

  void PrintConfigVals();

  uint64_t intVal(const std::string& key, uint32_t index = 0) const;

  double doubleVal(const std::string& key, uint32_t index = 0) const;

  float floatVal(const std::string& key, uint32_t index = 0) const;

  const std::string& strVal(const std::string& key, uint32_t index = 0) const;

  bool valExists(const std::string& key) const;

 private:
  void parseConfig(const std::string& filename);

  std::regex key_re, int_re, decimal_re, string_re, comment_re, empty_re;

  std::unordered_map<std::string, std::shared_ptr<ConfigValue>> config_vars;
};

}  // namespace thirdai::bolt

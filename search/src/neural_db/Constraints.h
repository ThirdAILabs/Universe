#pragma once

#include <cereal/access.hpp>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace thirdai::search::ndb {

enum class MetadataType { Bool, Int, Float, Str, Nil };

class MetadataValue {
 public:
  MetadataValue() : _type(MetadataType::Nil) {}

  static MetadataValue Bool(bool value) {
    return MetadataValue(MetadataType::Bool, value);
  }

  static MetadataValue Int(int value) {
    return MetadataValue(MetadataType::Int, value);
  }

  static MetadataValue Float(float value) {
    return MetadataValue(MetadataType::Float, value);
  }

  static MetadataValue Str(std::string value) {
    return MetadataValue(MetadataType::Str, std::move(value));
  }

  MetadataType type() const { return _type; }

  bool asBool() const {
    checkType(MetadataType::Bool);
    return std::get<bool>(_value);
  }

  int asInt() const {
    checkType(MetadataType::Int);
    return std::get<int>(_value);
  }

  float asFloat() const {
    checkType(MetadataType::Float);
    return std::get<float>(_value);
  }

  std::string asStr() const {
    checkType(MetadataType::Str);
    return std::get<std::string>(_value);
  }

  bool equals(const MetadataValue& other) const {
    return _type == other._type && _value == other._value;
  }

  bool lessThan(const MetadataValue& other) const {
    return _type == other._type && _value < other._value;
  }

  bool greaterThan(const MetadataValue& other) const {
    return _type == other._type && _value > other._value;
  }

 private:
  MetadataValue(MetadataType type,
                std::variant<bool, int, float, std::string> value)
      : _type(type), _value(std::move(value)) {}

  friend class cereal::access;

  template <class Archive>
  void save(Archive& archive) const {
    archive(_type);
    switch (_type) {
      case MetadataType::Bool:
        archive(std::get<bool>(_value));
        break;
      case MetadataType::Int:
        archive(std::get<int>(_value));
        break;
      case MetadataType::Float:
        archive(std::get<float>(_value));
        break;
      case MetadataType::Str:
        archive(std::get<std::string>(_value));
        break;
      case MetadataType::Nil:
        break;
    }
  }

  template <class Archive>
  void load(Archive& archive) {
    archive(_type);
    switch (_type) {
      case MetadataType::Bool:
        bool b;
        archive(b);
        _value = b;
        break;
      case MetadataType::Int:
        int i;
        archive(i);
        _value = i;
        break;
      case MetadataType::Float:
        float f;
        archive(f);
        _value = f;
        break;
      case MetadataType::Str:
        _value = loadStr(archive);
        break;
      case MetadataType::Nil:
        break;
    }
  }

  template <class Archive>
  std::string loadStr(Archive& archive) {
    std::string s;
    archive(s);
    return s;
  }

  void checkType(MetadataType expected) const {
    if (_type != expected) {
      throw std::runtime_error("cannot convert metadata type " +
                               typeToString(_type) + " to type " +
                               typeToString(expected));
    }
  }

  static std::string typeToString(MetadataType type) {
    switch (type) {
      case MetadataType::Bool:
        return "bool";
      case MetadataType::Int:
        return "int";
      case MetadataType::Float:
        return "float";
      case MetadataType::Str:
        return "str";
      case MetadataType::Nil:
        return "nil";
    }
  }

  MetadataType _type;

  std::variant<bool, int, float, std::string> _value;
};

using MetadataMap = std::unordered_map<std::string, MetadataValue>;

std::string serializeMetadata(const MetadataMap& metadata);

MetadataMap deserializeMetadata(const std::string& bytes);

class Constraint {
 public:
  virtual bool matches(const MetadataValue& value) const = 0;

  virtual ~Constraint() = default;
};

using QueryConstraints =
    std::unordered_map<std::string, std::shared_ptr<Constraint>>;

bool matches(const QueryConstraints& constraints, const MetadataMap& metadata);

class EqualTo final : public Constraint {
 public:
  explicit EqualTo(MetadataValue value) : _value(std::move(value)) {}

  static std::shared_ptr<EqualTo> make(MetadataValue value) {
    return std::make_shared<EqualTo>(std::move(value));
  }

  bool matches(const MetadataValue& value) const final {
    return _value.equals(value);
  }

 private:
  MetadataValue _value;
};

class AnyOf final : public Constraint {
 public:
  explicit AnyOf(std::vector<MetadataValue> values)
      : _values(std::move(values)) {}

  static std::shared_ptr<AnyOf> make(std::vector<MetadataValue> values) {
    return std::make_shared<AnyOf>(std::move(values));
  }

  bool matches(const MetadataValue& value) const final {
    return std::any_of(_values.begin(), _values.end(),
                       [&value](const auto& v) { return value.equals(v); });
  }

 private:
  std::vector<MetadataValue> _values;
};

class LessThan final : public Constraint {
 public:
  explicit LessThan(MetadataValue value) : _value(std::move(value)) {}

  static std::shared_ptr<LessThan> make(MetadataValue value) {
    return std::make_shared<LessThan>(std::move(value));
  }

  bool matches(const MetadataValue& value) const final {
    return value.lessThan(_value);
  }

 private:
  MetadataValue _value;
};

class GreaterThan final : public Constraint {
 public:
  explicit GreaterThan(MetadataValue value) : _value(std::move(value)) {}

  static std::shared_ptr<GreaterThan> make(MetadataValue value) {
    return std::make_shared<GreaterThan>(std::move(value));
  }

  bool matches(const MetadataValue& value) const final {
    return value.greaterThan(_value);
  }

 private:
  MetadataValue _value;
};

}  // namespace thirdai::search::ndb

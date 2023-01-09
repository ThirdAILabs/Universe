#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <cstdint>
#include <string>
#include <vector>

namespace thirdai::dataset {

struct ColumnIdentifier {
  ColumnIdentifier() {}

  // NOLINTNEXTLINE Ignore implicit conversion warning. That is intentional.
  ColumnIdentifier(uint32_t column_number) : _column_number(column_number) {}

  // NOLINTNEXTLINE Ignore implicit conversion warning. That is intentional.
  ColumnIdentifier(const std::string& column_name)
      : _column_name(column_name) {}

  bool hasName() const { return !!_column_name; }

  const std::string& name() const {
    if (!_column_name) {
      throw std::runtime_error(
          "Tried to get missing column name from ColumnIdentifier.");
    }
    return _column_name.value();
  }

  bool hasNumber() const { return _column_number.has_value(); }

  uint32_t number() const {
    if (!_column_number) {
      throw std::runtime_error(
          "Tried to get missing column number from ColumnIdentifier.");
    }
    return _column_number.value();
  }

  void updateColumnNumber(const ColumnNumberMap& column_number_map) {
    _column_number = column_number_map.at(name());
  }

  friend bool operator==(const ColumnIdentifier& lhs,
                         const ColumnIdentifier& rhs) {
    if (lhs.hasName() != rhs.hasName()) {
      return false;
    }
    if (lhs.hasName() && lhs.name() != rhs.name()) {
      return false;
    }
    if (lhs.hasNumber() != rhs.hasNumber()) {
      return false;
    }
    if (lhs.hasNumber() && lhs.number() != rhs.number()) {
      return false;
    }
    return true;
  }

 private:
  std::optional<uint32_t> _column_number;
  std::optional<std::string> _column_name;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_column_number, _column_name);
  }
};

}  // namespace thirdai::dataset
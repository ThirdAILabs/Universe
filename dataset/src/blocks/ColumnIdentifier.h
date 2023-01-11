#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::dataset {

struct ColumnIdentifier {
  ColumnIdentifier() {}

  // NOLINTNEXTLINE Ignore implicit conversion warning. That is intentional.
  ColumnIdentifier(uint32_t column_number)
      : _column_number(column_number), _column_name(std::nullopt) {}

  // NOLINTNEXTLINE Ignore implicit conversion warning. That is intentional.
  ColumnIdentifier(const std::string& column_name)
      : _column_number(std::nullopt), _column_name(column_name) {}

  bool consistentWith(const ColumnIdentifier& other) const {
    return hasName() == other.hasName() && hasNumber() == other.hasNumber();
  }

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
    if (!hasName()) {
      throw std::logic_error(
          "Cannot update the column number of a ColumnIdentifier that does not "
          "have a column name.");
    }
    _column_number = column_number_map.at(name());
  }

  friend bool operator==(const ColumnIdentifier& lhs,
                         const ColumnIdentifier& rhs) {
    return lhs._column_name == rhs._column_name &&
           lhs._column_number == rhs._column_number;
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
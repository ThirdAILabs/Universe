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

  ColumnIdentifier(const std::string& column_name, uint32_t column_number)
      : _column_number(column_number), _column_name(column_name) {}

  bool consistentWith(const ColumnIdentifier& other) const;

  bool hasName() const;

  const std::string& name() const;

  bool hasNumber() const;

  uint32_t number() const;

  void updateColumnNumber(const ColumnNumberMap& column_number_map);

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
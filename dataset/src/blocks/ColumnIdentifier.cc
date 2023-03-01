#include "ColumnIdentifier.h"

namespace thirdai::dataset {

bool ColumnIdentifier::consistentWith(const ColumnIdentifier& other) const {
  return hasName() == other.hasName() && hasNumber() == other.hasNumber();
}

bool ColumnIdentifier::hasName() const { return !!_column_name; }

const std::string& ColumnIdentifier::name() const {
  if (!_column_name) {
    throw std::runtime_error(
        "Tried to get missing column name from ColumnIdentifier.");
  }
  return _column_name.value();
}

bool ColumnIdentifier::hasNumber() const { return _column_number.has_value(); }

uint32_t ColumnIdentifier::number() const {
  if (!_column_number) {
    throw std::runtime_error(
        "Tried to get missing column number from ColumnIdentifier.");
  }
  return _column_number.value();
}

void ColumnIdentifier::updateColumnNumber(
    const ColumnNumberMap& column_number_map) {
  if (!hasName()) {
    throw std::logic_error(
        "Cannot update the column number of a ColumnIdentifier that does not "
        "have a column name.");
  }
  _column_number = column_number_map.at(name());
}

}  // namespace thirdai::dataset
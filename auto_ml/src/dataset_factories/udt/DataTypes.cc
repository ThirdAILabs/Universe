#include "DataTypes.h"

namespace thirdai::automl::data {

CategoricalDataTypePtr asCategorical(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<CategoricalDataType>(data_type);
}

NumericalDataTypePtr asNumerical(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<NumericalDataType>(data_type);
}

TextDataTypePtr asText(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<TextDataType>(data_type);
}

DateDataTypePtr asDate(const DataTypePtr& data_type) {
  return std::dynamic_pointer_cast<DateDataType>(data_type);
}

}  // namespace thirdai::automl::data

CEREAL_REGISTER_TYPE(thirdai::automl::data::CategoricalDataType)
CEREAL_REGISTER_TYPE(thirdai::automl::data::NumericalDataType)
CEREAL_REGISTER_TYPE(thirdai::automl::data::DateDataType)
CEREAL_REGISTER_TYPE(thirdai::automl::data::TextDataType)
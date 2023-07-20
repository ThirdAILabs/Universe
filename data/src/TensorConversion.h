#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <data/src/ColumnMap.h>

namespace thirdai::data {

using bolt::nn::tensor::TensorList;

std::vector<TensorList> convertToTensors(const ColumnMap& columns,
                                         const std::string& indices_column,
                                         const std::string& values_column,
                                         size_t batch_size);

}  // namespace thirdai::data
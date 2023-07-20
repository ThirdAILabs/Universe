#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <data/src/ColumnMap.h>

namespace thirdai::data {

using bolt::nn::tensor::TensorList;

using IndexValueColumnList = std::vector<std::pair<std::string, std::string>>;

std::vector<TensorList> convertToTensors(
    const ColumnMap& columns, const IndexValueColumnList& columns_to_convert,
    size_t batch_size);

}  // namespace thirdai::data
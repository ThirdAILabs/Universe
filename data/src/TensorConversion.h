#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <data/src/ColumnMap.h>

namespace thirdai::data {

using IndexValueColumnList =
    std::vector<std::pair<std::string, std::optional<std::string>>>;

std::vector<bolt::TensorList> toTensorBatches(
    const ColumnMap& columns, const IndexValueColumnList& columns_to_convert,
    size_t batch_size);

bolt::TensorList toTensors(const ColumnMap& columns,
                           const IndexValueColumnList& columns_to_convert);

}  // namespace thirdai::data
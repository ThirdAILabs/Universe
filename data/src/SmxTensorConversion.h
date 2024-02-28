#pragma once

#include "TensorConversion.h"
#include <smx/src/tensor/Tensor.h>

namespace thirdai::data {

using SmxDataset = std::vector<std::vector<smx::TensorPtr>>;

SmxDataset toSmxTensorBatches(const ColumnMap& columns,
                              const OutputColumnsList& columns_to_convert,
                              size_t batch_size);

std::vector<smx::TensorPtr> toSmxTensors(
    const ColumnMap& columns, const OutputColumnsList& columns_to_convert);

}  // namespace thirdai::data
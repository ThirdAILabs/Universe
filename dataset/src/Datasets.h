#pragma once

#include "BatchProcessor.h"
#include "InMemoryDataset.h"
#include "StreamingDataset.h"
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::dataset {

// TODO(Nicholas, Josh): Rename to just Dataset?
using BoltDataset = InMemoryDataset<BoltBatch>;
using BoltDatasetPtr = std::shared_ptr<BoltDataset>;
using BoltDatasetList = std::vector<BoltDatasetPtr>;

}  // namespace thirdai::dataset
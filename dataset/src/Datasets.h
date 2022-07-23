#pragma once

#include "BatchProcessor.h"
#include "InMemoryDataset.h"
#include "StreamingDataset.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>

namespace thirdai::dataset {

// TODO(Nicholas, Josh): Rename to just Dataset?
using BoltDataset = InMemoryDataset<bolt::BoltBatch>;
using BoltDatasetPtr = std::shared_ptr<BoltDataset>;
using BoltDatasetList = std::vector<BoltDatasetPtr>;

using BoltTokenDataset = InMemoryDataset<BoltTokenBatch>;
using BoltTokenDatasetPtr = std::shared_ptr<BoltTokenDataset>;
using BoltTokenDatasetList = std::vector<BoltTokenDatasetPtr>;

}  // namespace thirdai::dataset
#pragma once

#include "Featurizer.h"
#include "InMemoryDataset.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/DataSource.h>

namespace thirdai::dataset {

// TODO(Nicholas): Rename to just Dataset?
using BoltDataset = InMemoryDataset;
using BoltDatasetPtr = std::shared_ptr<BoltDataset>;
using BoltDatasetList = std::vector<BoltDatasetPtr>;

}  // namespace thirdai::dataset

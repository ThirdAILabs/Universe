#pragma once

#include "BlockInterface.h"

namespace thirdai::dataset {

class TabularMetadata {};

class TabularPairGram : public Block {
  explicit TabularPairGram(TabularMetadata metadata) : _metadata(metadata) {}

 private:
  TabularMetadata _metadata;
};

class TabularLabel : public Block {
  explicit TabularLabel(TabularMetadata metadata) : _metadata(metadata) {}

 private:
  TabularMetadata _metadata;
};

}  // namespace thirdai::dataset

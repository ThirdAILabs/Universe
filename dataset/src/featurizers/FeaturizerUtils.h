#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/BlockInterface.h>
namespace thirdai::dataset {

std::exception_ptr buildVector(BoltVector& vector, BlockList& blocks,
                               ColumnarInputSample& sample,
                               std::optional<uint32_t> hash_range);

std::shared_ptr<SegmentedFeatureVector> makeSegmentedFeatureVector(
    bool blocks_dense, std::optional<uint32_t> hash_range,
    bool store_segment_feature_map);

}  // namespace thirdai::dataset
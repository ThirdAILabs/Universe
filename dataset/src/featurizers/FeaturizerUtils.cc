#include <dataset/src/featurizers/FeaturizerUtils.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>

namespace thirdai::dataset {

std::exception_ptr buildVector(BoltVector& vector, BlockList& blocks,
                               ColumnarInputSample& sample,
                               std::optional<uint32_t> hash_range) {
  auto segmented_vector =
      makeSegmentedFeatureVector(blocks.areDense(), hash_range,
                                 /* store_segment_feature_map= */ false);
  if (auto err = blocks.addVectorSegments(sample, *segmented_vector)) {
    return err;
  }
  vector = segmented_vector->toBoltVector();
  return nullptr;
}

std::shared_ptr<SegmentedFeatureVector> makeSegmentedFeatureVector(
    bool blocks_dense, std::optional<uint32_t> hash_range,
    bool store_segment_feature_map) {
  if (hash_range) {
    return std::make_shared<HashedSegmentedFeatureVector>(
        *hash_range, store_segment_feature_map);
  }
  // Dense vector if all blocks produce dense features, sparse vector
  // otherwise.
  if (blocks_dense) {
    return std::make_shared<SegmentedDenseFeatureVector>(
        store_segment_feature_map);
  }
  return std::make_shared<SegmentedSparseFeatureVector>(
      store_segment_feature_map);
}

}  // namespace thirdai::dataset
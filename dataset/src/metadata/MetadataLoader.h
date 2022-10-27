#include "Metadata.h"
#include <dataset/src/DataLoader.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <cstdint>
#include <vector>

namespace thirdai::dataset {

class MetadataLoader {
 public:
  static auto loadMetadata(DataLoaderPtr loader,
                           std::vector<BlockPtr> feature_blocks,
                           uint32_t key_col, uint32_t n_unique_keys,
                           char delimiter = ',') {
    auto label_block =
        StringLookupCategoricalBlock::make(key_col, n_unique_keys);
    auto key_vocab = label_block->getVocabulary();

    StreamingGenericDatasetLoader metadataset(
        std::move(loader),
        /* input_blocks= */ std::move(feature_blocks),
        /* label_blocks= */ {label_block}, /* shuffle= */ false,
        DatasetShuffleConfig(), /* has_header= */ false,
        /* delimiter= */ delimiter);

    auto dim = metadataset.getInputDim();

    auto [vectors, key_ids] = metadataset.loadInMemory();
    key_vocab->fixVocab();

    auto vectors_map = buildVectorMap(*vectors, *key_ids);

    return Metadata::make(std::move(vectors_map), std::move(key_vocab), dim);
  }

 private:
  static std::vector<BoltVector> buildVectorMap(BoltDataset& vectors,
                                                const BoltDataset& key_ids) {
    std::vector<BoltVector> vectors_map(vectors.len());

    for (uint32_t batch_idx = 0; batch_idx < vectors.numBatches();
         batch_idx++) {
      auto& vector_batch = vectors.at(batch_idx);
      const auto& key_id_batch = key_ids.at(batch_idx);
      for (uint32_t vec_idx = 0; vec_idx < vector_batch.getBatchSize();
           vec_idx++) {
        auto& preprocessed_vector = vector_batch[vec_idx];
        const auto& uid = key_id_batch[vec_idx].active_neurons[0];
        vectors_map.at(uid) = std::move(preprocessed_vector);
      }
    }
    return vectors_map;
  }
};

}  // namespace thirdai::dataset
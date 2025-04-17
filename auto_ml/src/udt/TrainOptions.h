#pragma once

#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <optional>

namespace thirdai::automl::udt {

struct TrainOptions {
  std::optional<size_t> batch_size = std::nullopt;
  std::optional<size_t> max_in_memory_batches = std::nullopt;
  std::optional<uint32_t> steps_per_validation = std::nullopt;
  bool sparse_validation = false;
  bool verbose = true;
  std::optional<uint32_t> logging_interval = std::nullopt;
  dataset::DatasetShuffleConfig shuffle_config =
      dataset::DatasetShuffleConfig();

  size_t batchSize() const { return batch_size.value_or(defaults::BATCH_SIZE); }
};

}  // namespace thirdai::automl::udt

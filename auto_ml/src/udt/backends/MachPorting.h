#pragma once

#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <dataset/src/blocks/text/Text.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::automl::udt {

struct MachInfo {
  std::shared_ptr<utils::Classifier> classifier;

  dataset::TextBlockPtr text_block;
  uint32_t feature_hash_range;

  dataset::mach::MachIndexPtr mach_index;

  std::string text_column_name;
  std::string label_column_name;
  std::optional<char> label_delimiter;

  char csv_delimiter;

  uint32_t default_top_k_to_return;
  uint32_t num_buckets_to_eval;
  float mach_sampling_threshold;

  std::optional<RLHFSampler> balancing_samples;
};

}  // namespace thirdai::automl::udt
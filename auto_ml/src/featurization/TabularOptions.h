#pragma once

#include <cereal/access.hpp>
#include <auto_ml/src/udt/Defaults.h>
#include <string>

namespace thirdai::automl::data {

struct TabularOptions {
  uint32_t text_pairgrams_word_limit = udt::defaults::PAIRGRAM_WORD_LIMIT;
  bool contextual_columns = udt::defaults::CONTEXTUAL_COLUMNS;
  std::string time_granularity = udt::defaults::TIME_GRANULARITY;
  uint32_t lookahead = udt::defaults::LOOKAHEAD;
  uint32_t feature_hash_range = udt::defaults::FEATURE_HASH_RANGE;
  char delimiter = udt::defaults::CSV_DELIMITER;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(text_pairgrams_word_limit, contextual_columns, time_granularity,
            lookahead, feature_hash_range, delimiter);
  }
};

}  // namespace thirdai::automl::data
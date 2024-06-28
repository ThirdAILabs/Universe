#pragma once

#include <search/src/inverted_index/Tokenizer.h>

namespace thirdai::search {

struct IndexConfig {
  size_t max_docs_to_score = 10000;
  float max_token_occurrence_frac = 0.2;

  // The k1 and b defaults are the same as the defaults for BM25 in apache
  // Lucene. The idf_cutoff_frac default is just what seemed to work fairly
  // well in multiple experiments.
  float k1 = 1.2;
  float b = 0.75;

  size_t shard_size = 10000000;

  TokenizerPtr tokenizer = std::make_shared<DefaultTokenizer>();
  std::string db_adapter = "mongodb";
};

}  // namespace thirdai::search
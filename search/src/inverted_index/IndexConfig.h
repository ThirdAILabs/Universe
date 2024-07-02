#pragma once

#include <search/src/inverted_index/Tokenizer.h>

#include <utility>

namespace thirdai::search {

struct IndexConfig {
    size_t max_docs_to_score;
    float max_token_occurrence_frac;
    // The k1 and b defaults are the same as the defaults for BM25 in apache
    // Lucene. The idf_cutoff_frac default is just what seemed to work fairly
    // well in multiple experiments.
    float k1;
    float b;
    size_t shard_size;
    TokenizerPtr tokenizer;
    std::string db_adapter;
    std::string db_uri;
    uint32_t batch_size;

    // Constructor for general defaults with rocksdb adapter
    IndexConfig() 
        : max_docs_to_score(10000),
          max_token_occurrence_frac(0.2),
          k1(1.2),
          b(0.75),
          shard_size(10000000),
          tokenizer(std::make_shared<DefaultTokenizer>()),
          db_adapter("rocksdb"),
          batch_size(0)  // Not used in rocksdb configuration
    {}

    // Constructor for MongoDB with specific uri and batch size
    IndexConfig(std::string uri, uint32_t bulk_update_batch=64000)
        : max_docs_to_score(10000),
          max_token_occurrence_frac(0.2),
          k1(1.2),
          b(0.75),
          shard_size(10000000),
          tokenizer(std::make_shared<DefaultTokenizer>()),
          db_adapter("mongodb"),
          db_uri(std::move(uri)),
          batch_size(bulk_update_batch)
    {}
};

}  // namespace thirdai::search
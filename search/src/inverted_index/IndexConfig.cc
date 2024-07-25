
#include "IndexConfig.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>

namespace thirdai::search {

ar::ConstArchivePtr IndexConfig::toArchive() const {
  auto map = ar::Map::make();

  map->set("max_docs_to_score", ar::u64(max_docs_to_score));
  map->set("max_token_occurrence_frac", ar::f32(max_token_occurrence_frac));
  map->set("k1", ar::f32(k1));
  map->set("b", ar::f32(b));
  map->set("shard_size", ar::u64(shard_size));

  map->set("tokenizer", tokenizer->toArchive());

  return map;
}

IndexConfig IndexConfig::fromArchive(const ar::Archive& archive) {
  IndexConfig config;

  config.max_docs_to_score = archive.u64("max_docs_to_score");
  config.max_token_occurrence_frac = archive.f32("max_token_occurrence_frac");
  config.k1 = archive.f32("k1");
  config.b = archive.f32("b");
  config.shard_size = archive.u64("shard_size");
  config.tokenizer = Tokenizer::fromArchive(*archive.get("tokenizer"));

  return config;
}

}  // namespace thirdai::search
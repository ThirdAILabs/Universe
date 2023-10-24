#include "FlashConfig.h"
#include "Parameter.h"
#include <hashing/src/DensifiedMinHash.h>
#include <hashing/src/HashFunction.h>
#include <hashing/src/MinHash.h>
#include <utils/StringManipulation.h>

namespace thirdai::automl::config {

std::shared_ptr<hashing::HashFunction> getHashFunction(
    const std::string& name, uint32_t num_tables, uint32_t hashes_per_table,
    uint32_t range) {
  if (text::lower(name) == "minhash") {
    return std::make_shared<hashing::MinHash>(hashes_per_table, num_tables,
                                              range);
  }
  if (text::lower(name) == "densifiedminhash") {
    return std::make_shared<hashing::DensifiedMinHash>(hashes_per_table,
                                                       num_tables, range);
  }

  throw exceptions::NotImplemented(
      "Unsupported Hash Function. Supported Hash Functions: "
      "DensifiedMinHash, MinHash.");
}

std::unique_ptr<search::Flash> buildIndex(const json& config,
                                                    const ArgumentMap& args) {
  uint32_t num_tables = integerParameter(config, "num_tables", args);
  uint32_t hashes_per_table =
      integerParameter(config, "hashes_per_table", args);
  uint32_t range = integerParameter(config, "range", args);

  std::string hash_fn_name = stringParameter(config, "hash_fn", args);

  auto hash_fn =
      getHashFunction(hash_fn_name, num_tables, hashes_per_table, range);

  if (config.contains("reservoir_size")) {
    uint32_t reservoir_size = integerParameter(config, "reservoir_size", args);

    return std::make_unique<search::Flash>(hash_fn, reservoir_size);
  }

  return std::make_unique<search::Flash>(hash_fn);
}

}  // namespace thirdai::automl::config
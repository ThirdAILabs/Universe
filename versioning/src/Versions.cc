#include "Versions.h"
#include <stdexcept>

namespace thirdai::versions {

void checkVersion(const uint32_t loaded_version, const uint32_t current_version,
                  const std::string& loaded_thirdai_version,
                  const std::string& current_thirdai_version,
                  const std::string& class_name) {
  if (loaded_version != current_version) {
#if THIRDAI_EXPOSE_ALL
    throw std::invalid_argument("Incompatible version. Expected version " +
                                std::to_string(current_version) + " for " +
                                class_name + ", but got version " +
                                std::to_string(loaded_version));
#endif
    (void)class_name;

    uint32_t current_thirdai_hash_pos = current_thirdai_version.find('+');
    std::string current_thirdai_short_version =
        current_thirdai_version.substr(0, current_thirdai_hash_pos);

    uint32_t loaded_thirdai_hash_pos = loaded_thirdai_version.find('+');
    std::string loaded_thirdai_short_version =
        loaded_thirdai_version.substr(0, loaded_thirdai_hash_pos);

    throw std::invalid_argument(
        "The model you are loading is not compatible with the current version "
        "of thirdai (v" +
        current_thirdai_short_version +
        "). Please downgrade to the version the model was saved with (v" +
        loaded_thirdai_short_version + ")");
  }
}

}  // namespace thirdai::versions
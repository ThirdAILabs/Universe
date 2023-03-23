
#include <exceptions/src/Exceptions.h>
#include <utils/StringManipulation.h>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <variant>

namespace thirdai::licensing {

const std::string FULL_ACCESS_ENTITLEMENT = "FULL_ACCESS";

const std::string FULL_MODEL_ENTITLEMENT = "FULL_MODEL_ACCESS";
const std::string FULL_DATASET_ENTITLEMENT = "FULL_DATASET_ACCESS";

const std::string LOAD_SAVE_ENTITLEMENT = "LOAD_SAVE";
const std::string MAX_TRAIN_SAMPLES_ENTITLEMENT_START = "MAX_TRAIN_SAMPLES";
const std::string MAX_OUTPUT_DIM_ENTITLEMENT_START = "MAX_OUTPUT_DIM";

struct FinegrainedModelAccess {
  explicit FinegrainedModelAccess(
      const std::unordered_set<std::string>& entitlement_strings);

  bool load_save;
  uint64_t max_train_samples;
  uint64_t max_output_dim;
};

struct FullModelAccess {};

struct FinegrainedDatasetAccess {
  explicit FinegrainedDatasetAccess(
      const std::unordered_set<std::string>& entitlement_strings);

  std::unordered_set<std::string> dataset_hashes;
};

struct FullDatasetAccess {};

struct FinegrainedFullAccess {
  explicit FinegrainedFullAccess(
      const std::unordered_set<std::string>& entitlement_strings);

  std::variant<FullModelAccess, FinegrainedModelAccess> model_access;
  std::variant<FullDatasetAccess, FinegrainedDatasetAccess> dataset_access;
};

struct FullAccess {};

struct EntitlementTree {
  std::variant<FullAccess, FinegrainedFullAccess> access;

  explicit EntitlementTree(
      const std::unordered_set<std::string>& entitlement_strings);
};

}  // namespace thirdai::licensing
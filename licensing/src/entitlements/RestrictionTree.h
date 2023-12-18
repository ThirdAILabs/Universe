#pragma once

#include <exceptions/src/Exceptions.h>
#include <utils/text/StringManipulation.h>
#include <cstdint>
#include <optional>
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

struct ModelRestrictions {
  explicit ModelRestrictions(
      const std::unordered_set<std::string>& entitlement_strings);

  bool can_load_save;
  uint64_t max_train_samples;
  uint64_t max_output_dim;

  // Util variable representing how many fields from entitlement_strings were
  // parsed in the creation of the struct
  uint32_t num_fields_parsed;
};

struct DatasetRestrictions {
  explicit DatasetRestrictions(
      const std::unordered_set<std::string>& entitlement_strings);

  bool datasetAllowed(const std::string& dataset_hash) {
    return _allowed_dataset_hashes.count(dataset_hash);
  }

  const std::unordered_set<std::string>& allowedDatasetHashes() {
    return _allowed_dataset_hashes;
  }

 private:
  std::unordered_set<std::string> _allowed_dataset_hashes;
};

struct Restrictions {
  explicit Restrictions(
      const std::unordered_set<std::string>& entitlement_strings);

  std::optional<ModelRestrictions> model_restrictions;
  std::optional<DatasetRestrictions> dataset_restrictions;
};

/**
 * We parse the entitlements, which represent a set of "allowed" actions,
 * into a RestrictionTree, which encode the restrictions placed on the
 * user. Both are representations of the same space of "allowed" actions, but
 * parsing into a restriction tree allows us to more easily verify that the
 * entitlements are in a correct configuration, and makes the data structure
 * overall cleaner. Unlike the original entitlements, the RestrictionTree is a
 * "negative" licensing system. Instead of users having entitlements that give
 * them additional access to the system, users have optional restrictions that
 * remove access. When doing the parsing, we ensure that if an entitlement is
 * not present that it gets transformed into the corresponding maximum
 * restriction.
 */
struct RestrictionTree {
  std::optional<Restrictions> restrictions;

  explicit RestrictionTree(
      const std::unordered_set<std::string>& entitlement_strings);
};

}  // namespace thirdai::licensing
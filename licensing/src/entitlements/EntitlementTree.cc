#include "EntitlementTree.h"
#include <iostream>

namespace thirdai::licensing {

FinegrainedFullAccess::FinegrainedFullAccess(
    const std::unordered_set<std::string>& entitlement_strings) {
  uint64_t fields_parsed = 0;

  if (entitlement_strings.count(FULL_MODEL_ENTITLEMENT)) {
    model_access = FullModelAccess();
    fields_parsed++;
  } else {
    model_access = FinegrainedModelAccess(entitlement_strings);
    fields_parsed += 3;
  }

  if (entitlement_strings.count(FULL_DATASET_ENTITLEMENT)) {
    dataset_access = FullDatasetAccess();
    fields_parsed++;
  } else {
    dataset_access = FinegrainedDatasetAccess(entitlement_strings);
    fields_parsed += std::get<FinegrainedDatasetAccess>(dataset_access)
                         .dataset_hashes.size();
  }

  // If there are more fields than we parsed, then this is potentially a new
  // license with restriction we don't handle, so we throw an exception to be
  // safe
  if (entitlement_strings.size() > fields_parsed) {
    throw exceptions::LicenseCheckException(
        "License could not be parsed because it contained unknown fields. "
        "Try upgrading your thirdai version.");
  }
}

FinegrainedDatasetAccess::FinegrainedDatasetAccess(
    const std::unordered_set<std::string>& entitlement_strings) {
  // For now, we just add all strings of length 64 (this will add all
  // sha256 hashes in the entitlements, as well as any other strings of
  // length 64).
  for (const auto& entitlement : entitlement_strings) {
    if (entitlement.size() == 64) {
      dataset_hashes.insert(entitlement);
    }
  }
}

FinegrainedModelAccess::FinegrainedModelAccess(
    const std::unordered_set<std::string>& entitlement_strings)
    : load_save(false), max_train_samples(0), max_output_dim(0) {
  if (entitlement_strings.count(LOAD_SAVE_ENTITLEMENT)) {
    load_save = true;
  }

  for (const auto& entitlement : entitlement_strings) {
    auto split_entitlement = text::split(entitlement);
    if (text::startsWith(entitlement, MAX_TRAIN_SAMPLES_ENTITLEMENT_START)) {
      if (split_entitlement.size() != 2) {
        throw exceptions::LicenseCheckException(
            "Invalid format of entitlement " + entitlement);
      }
      max_train_samples = std::stoul(std::string(split_entitlement.at(1)));
    }
    if (text::startsWith(entitlement, MAX_OUTPUT_DIM_ENTITLEMENT_START)) {
      if (split_entitlement.size() != 2) {
        throw exceptions::LicenseCheckException(
            "Invalid format of entitlement " + entitlement);
      }
      max_output_dim = std::stoul(std::string(split_entitlement.at(1)));
    }
  }
}

EntitlementTree::EntitlementTree(
    const std::unordered_set<std::string>& entitlement_strings) {
  if (entitlement_strings.count(FULL_ACCESS_ENTITLEMENT)) {
    access = FullAccess();
    return;
  }

  access = FinegrainedFullAccess(entitlement_strings);
}

}  // namespace thirdai::licensing
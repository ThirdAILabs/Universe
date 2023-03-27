#include "RestrictionTree.h"
#include <iostream>
#include <optional>

namespace thirdai::licensing {

Restrictions::Restrictions(
    const std::unordered_set<std::string>& entitlement_strings) {
  uint64_t fields_parsed = 0;

  if (entitlement_strings.count(FULL_MODEL_ENTITLEMENT)) {
    model_restrictions = std::nullopt;
    fields_parsed++;
  } else {
    model_restrictions = ModelRestrictions(entitlement_strings);
    fields_parsed += 3;
  }

  if (entitlement_strings.count(FULL_DATASET_ENTITLEMENT)) {
    dataset_restrictions = std::nullopt;
    fields_parsed++;
  } else {
    dataset_restrictions = DatasetRestrictions(entitlement_strings);
    fields_parsed += dataset_restrictions->allowedDatasetHashes().size();
  }

  // If there are more fields than we parsed, then this is potentially a new
  // license with restriction we don't handle, so we throw an exception to be
  // safe. This check may miss a few edge cases, since it's possible that we
  // will have counted a string of length 64 that is not a dataset hash, and
  // so our fields_parsed variable will be larger than it should e.
  if (entitlement_strings.size() > fields_parsed) {
    throw exceptions::LicenseCheckException(
        "License could not be parsed because it contained unknown restrictions "
        "or contained known restrictions in an unsupported configuration. "
        "Try upgrading your thirdai version and make sure that your license is "
        "correct.");
  }
}

DatasetRestrictions::DatasetRestrictions(
    const std::unordered_set<std::string>& entitlement_strings) {
  // For now, we just add all strings of length 64 (this will add all
  // sha256 hashes in the entitlements, as well as any other strings of
  // length 64). This is not a problem because it's fine if we add strings to
  // the set that are not valid hashes because we only care that the dataset
  // hash is a part of the set.
  for (const auto& entitlement : entitlement_strings) {
    if (entitlement.size() == 64) {
      _allowed_dataset_hashes.insert(entitlement);
    }
  }
}

ModelRestrictions::ModelRestrictions(
    const std::unordered_set<std::string>& entitlement_strings)
    : can_load_save(false), max_train_samples(0), max_output_dim(0) {
  if (entitlement_strings.count(LOAD_SAVE_ENTITLEMENT)) {
    can_load_save = true;
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

RestrictionTree::RestrictionTree(
    const std::unordered_set<std::string>& entitlement_strings) {
  if (entitlement_strings.count(FULL_ACCESS_ENTITLEMENT)) {
    restrictions = std::nullopt;
    return;
  }

  restrictions = Restrictions(entitlement_strings);
}

}  // namespace thirdai::licensing
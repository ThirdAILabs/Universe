#pragma once
#include <dataset/src/DataLoader.h>
#include <optional>
#include <string>

namespace thirdai::licensing {

void checkLicense();

void setLicensePath(const std::string& license_path);

void activate(const std::string& api_key);

void deactivate();

bool isPartial();

void verifyAllowedDataset(const dataset::DataLoaderPtr& data_loader);

}  // namespace thirdai::licensing
#pragma once
#include <dataset/src/DataSource.h>
#include <optional>
#include <string>

namespace thirdai::licensing {

void checkLicense();

void setLicensePath(const std::string& license_path);

void activate(const std::string& api_key);

void deactivate();

void verifyAllowedDataset(const dataset::DataSourcePtr& data_source);

}  // namespace thirdai::licensing
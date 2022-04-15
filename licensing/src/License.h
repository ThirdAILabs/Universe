#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <utility>

namespace thirdai::licensing {

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

class License {
 public:
  License() {}

  License(std::unordered_map<std::string, std::string> metadata,
          int64_t expire_time_epoch_millis)
      : expire_time_epoch_millis(expire_time_epoch_millis),
        metadata(std::move(metadata)) {}

  static License createLicenseWithNDaysLeft(
      std::unordered_map<std::string, std::string> metadata, int64_t num_days) {
    int64_t current_millis = getCurrentEpochMillis();
    int64_t millis_offset = num_days * 24 * 3600 * 1000;
    int64_t expire_time = current_millis + millis_offset;
    return License(std::move(metadata), expire_time);
  }

  bool isExpired() const {
    return expire_time_epoch_millis < getCurrentEpochMillis();
  }

  std::string getMetadataValue(const std::string& key) const {
    return metadata.at(key);
  }

  int64_t getExpireTimeMillis() const { return expire_time_epoch_millis; }

  // Gets a string that represents the state of the license. This is the state
  // that is signed by our private key and later verified by the public key.
  std::string toString() {
    std::string to_verify;
    to_verify += std::to_string(expire_time_epoch_millis);
    to_verify += "|";
    for (auto const& [key, val] : metadata) {
      to_verify += key;
      to_verify += ":";
      to_verify += val;
    }
    return to_verify;
  }

 private:
  static int64_t getCurrentEpochMillis() {
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
        .count();
  }

  int64_t expire_time_epoch_millis;
  std::unordered_map<std::string, std::string> metadata;
};

}  // namespace thirdai::licensing
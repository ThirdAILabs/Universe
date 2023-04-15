#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <licensing/src/entitlements/Entitlements.h>
#include <chrono>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>

namespace thirdai::licensing {

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

class License {
 public:
  License(std::map<std::string, std::string> entitlements,
          int64_t expire_time_epoch_millis)
      : _expire_time_epoch_millis(expire_time_epoch_millis),
        _start_time_epoch_millis(getCurrentEpochMillis()),
        _entitlements(std::move(entitlements)) {}

  static License createLicenseWithNDaysLeft(
      std::map<std::string, std::string> entitlements, int64_t num_days) {
    int64_t current_millis = getCurrentEpochMillis();
    int64_t millis_offset = num_days * 24 * 3600 * 1000;
    int64_t expire_time = current_millis + millis_offset;
    return {std::move(entitlements), expire_time};
  }

  bool isExpired() const {
    return _start_time_epoch_millis > getCurrentEpochMillis() ||
           _expire_time_epoch_millis < getCurrentEpochMillis();
  }

  int64_t getExpireTimeMillis() const { return _expire_time_epoch_millis; }

  // Gets a string that represents the state of the license. This is the state
  // that is signed by our private key and later verified by the public key.
  std::string toString() const {
    std::string to_verify;
    to_verify += std::to_string(_expire_time_epoch_millis);
    to_verify += "|";
    for (auto const& [key, val] : _entitlements) {
      to_verify += key;
      to_verify += ":";
      to_verify += val;
      to_verify += ",";
    }
    return to_verify;
  }

  Entitlements entitlements() const {
    std::unordered_set<std::string> entitlements;
    for (const auto& key : _entitlements) {
      entitlements.insert(key.second);
    }
    return Entitlements(entitlements);
  }

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_start_time_epoch_millis, _expire_time_epoch_millis, _entitlements);
  }

  License();

  static int64_t getCurrentEpochMillis() {
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
        .count();
  }

  int64_t _expire_time_epoch_millis;
  int64_t _start_time_epoch_millis;
  // This is a map rather than an unordered map because when creating
  // the string to verify, we want to be easily able to generate a deterministic
  // string from the map (and unordered maps have non deterministic orders)
  // TODO(Josh): Consider making this an Entitlements object instead of this
  // map, since only the keys are ever used anyways.
  std::map<std::string, std::string> _entitlements;
};

}  // namespace thirdai::licensing
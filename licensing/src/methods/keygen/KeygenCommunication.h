#pragma once

#include <string>
#include <unordered_set>

namespace thirdai::licensing {

/*
 * Communicates with the Keygen server to verify that the user with the given
 * access key is validated to use the ThirdAI python package. Returns the
 * set of entitlements that the access_key has.
 */
std::unordered_set<std::string> verifyWithKeygen(const std::string& access_key);

}  // namespace thirdai::licensing
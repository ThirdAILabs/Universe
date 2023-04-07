#pragma once

#include <licensing/src/entitlements/Entitlements.h>
#include <string>
#include <unordered_set>

namespace thirdai::licensing::keygen {

/*
 * Communicates with the Keygen server to verify that the user with the given
 * access key is validated to use the ThirdAI python package. Returns the
 * set of entitlements that the access_key has.
 */
Entitlements entitlementsFromKeygen(const std::string& access_key);

}  // namespace thirdai::licensing::keygen
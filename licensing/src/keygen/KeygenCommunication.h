#pragma once

#include <string>

namespace thirdai::licensing {

  /*
   * Communicates with the Keygen server to verify that the user with the given
   * access key is validated to use the ThirdAI python package.
   */
  void verifyWithKeygen(const std::string& access_key);

}  // namespace thirdai::licensing
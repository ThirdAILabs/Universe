#pragma once

#include <string>

namespace thirdai::licensing {

class KeygenCommunication {
 public:
  /*
   * Communicates with the Keygen server to verify that the user with the given
   * access key is validated to use the ThirdAI python package.
   */
  static void verifyWithKeygen(const std::string& access_key);
};

}  // namespace thirdai::licensing
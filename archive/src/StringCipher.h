#pragma once

#include <string>

namespace thirdai::ar {

/**
 * This is to make strings that we save in our binary archives not human
 * readable, to avoid users gleaning architecture details just from looking at
 * keywords in the binary, for instance "fully_connected", "dim", etc.
 */
inline std::string cipher(const std::string& key) {
  // os.urandom(8).hex()
  const uint8_t keys[8] = {0x23, 0xbf, 0x35, 0xe9, 0x14, 0xd6, 0x88, 0x42};

  std::string out;
  for (size_t i = 0; i < key.size(); i++) {
    out.push_back(key[i] ^ keys[i % sizeof(keys)]);
  }

  return out;
}

}  // namespace thirdai::ar
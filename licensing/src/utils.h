#include <cryptopp/sha.h>  // SHA256
#include <string>

namespace thirdai::licensing {

/*
 * This method returns a SHA256 hash on the input string.
 * See https://www.cryptopp.com/wiki/SHA2 for more details.
 */
inline std::string sha256(const std::string& input) {
  std::string digest;
  CryptoPP::SHA256 hash;

  hash.Update(reinterpret_cast<const CryptoPP::byte*>(input.data()),
              input.length());
  digest.resize(hash.DigestSize());
  hash.Final(reinterpret_cast<CryptoPP::byte*>(digest.data()));

  return digest;
}

}  // namespace thirdai::licensing

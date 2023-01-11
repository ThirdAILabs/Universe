#pragma once

#include <cryptopp/base64.h>
#include <cryptopp/files.h>    // FileSource
#include <cryptopp/filters.h>  // HashFilter
#include <cryptopp/hex.h>      // HexEncoder
#include <cryptopp/sha.h>      // SHA256
#include <cryptopp/xed25519.h>
#include <random>
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

inline std::string sha256File(const std::string& filename) {
  CryptoPP::SHA256 hash;
  std::string digest;

  // Verbatim from https://www.cryptopp.com/wiki/Hash_Functions
  CryptoPP::FileSource f(
      /* filename = */ filename.data(), /* pumpAll = */ true,
      new CryptoPP::HashFilter(
          hash, new CryptoPP::HexEncoder(new CryptoPP::StringSink(
                    /* output = */ digest))));

  return digest;
}

inline std::string getRandomIdentifier(uint32_t numBytesRandomness) {
  static std::random_device dev;
  static std::mt19937 rng(dev());

  const char* v = "0123456789abcdef";
  std::uniform_int_distribution<uint32_t> dist(0, strlen(v));

  std::string res;
  for (uint32_t i = 0; i < numBytesRandomness * 2; i++) {
    res += v[dist(rng)];
  }
  return res;
}

/*
 * Creates an ed25519Verifier from the passed in public key string. The string
 * should be a DER format file encoded in base64.
 */
inline CryptoPP::ed25519Verifier createVerifierFromBase64String(
    const std::string& base64_key) {
  CryptoPP::StringSource source(/* string = */ base64_key,
                                /* pumpAll = */ true,
                                /* attachment = */ new CryptoPP::Base64Decoder);
  return CryptoPP::ed25519::Verifier(source);
}

}  // namespace thirdai::licensing

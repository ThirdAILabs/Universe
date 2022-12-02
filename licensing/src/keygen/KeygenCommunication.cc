#include <cryptopp/hex.h>
#include <stdexcept>
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "KeygenCommunication.h"
#include "cryptopp/sha.h"       // SHA256
#include "cryptopp/xed25519.h"  // Ed25519
#include "httplib.h"
#include "json.hpp"
#include <cryptopp/base64.h>  // Base64 decoder
#include <iostream>

namespace thirdai::licensing {

using json = nlohmann::json;

std::string get_signature(const std::string& signature_header) {
  // We add 11 because the actual signature starts 11 chars after the beginning
  // of the found string
  size_t signature_start = signature_header.find("signature=") + 11; 
  // The signature ends one char before the next " after signature_start
  size_t signature_end = signature_header.find(',', signature_start) - 1;

  return signature_header.substr(signature_start, signature_end - signature_start);
}

void verifySignature(const httplib::Result& res) {
  if (!res->has_header("Keygen-Signature")) {
    throw std::runtime_error(
        "License was found to be valid, but did not find a Keygen signature "
        "verifying that the response came from Keygen.");
  }

  if (!res->has_header("date")) {
    throw std::runtime_error(
        "License was found to be valid, but did not find a date in the"
        "response (necessary to verify that the response came from Keygen).");
  }

  std::string signature = res->get_header_value("Keygen-Signature");
  std::cout << signature << std::endl;
  signature = get_signature(signature);
  std::cout << signature << std::endl;

  // See https://www.cryptopp.com/wiki/SHA2
  std::string digest;
  CryptoPP::SHA256 hash;

  hash.Update(reinterpret_cast<const CryptoPP::byte*>(res->body.data()),
              res->body.length());
  digest.resize(hash.DigestSize());
  hash.Final(reinterpret_cast<CryptoPP::byte*>(&digest[0]));

  std::string encoded_digest;
  CryptoPP::StringSource(digest, true, new CryptoPP::Base64Encoder( // NOLINT
          new CryptoPP::StringSink(encoded_digest))  // Base64Encoder
  ); 

  std::stringstream signed_data_stream;
  signed_data_stream << "(request-target): post /v1/accounts/thirdai/licenses/actions/validate-key\n"
         << "host: api.keygen.sh\n"
         << "date: " << res->get_header_value("date") << "\n"
         << "digest: sha-256=" << encoded_digest;
  std::string signed_data = signed_data_stream.str();

  std::string BASE64_VERIFY_PUBLIC_KEY_DER =
      "MCowBQYDK2VwAyEAmtv9iB02PTHBVsNImWiS3QGDp+RUDcABy3wu7Fp5Zq4=";
  CryptoPP::ed25519::Verifier verifier;
  CryptoPP::StringSource ss(BASE64_VERIFY_PUBLIC_KEY_DER, true,
                            new CryptoPP::Base64Decoder);
  verifier.AccessPublicKey().Load(ss);

  std::cout << signed_data << std::endl;
  std::cout << signature << std::endl;
  bool valid = verifier.VerifyMessage(
      reinterpret_cast<const CryptoPP::byte*>(&signed_data[0]),
      signed_data.size(),
      reinterpret_cast<const CryptoPP::byte*>(&signature[0]), signature.size());

  if (valid == false) {
    throw std::runtime_error("Invalid signature over message");
  }
}

void KeygenCommunication::verifyWithKeygen(const std::string& access_key) {
  httplib::Client cli("https://api.keygen.sh");
  httplib::Headers headers = {{"Content-Type", "application/vnd.api+json"},
                              {"Accept", "application/vnd.api+json"}};
  json body;
  body["meta"]["key"] = access_key;

  httplib::Result res = cli.Post(
      /* path = */ "/v1/accounts/thirdai/licenses/actions/validate-key",
      /* headers = */ headers,
      /* body = */ body.dump(), /* content_type = */ "application/json");

  if (!res) {
    throw std::runtime_error("Licensing check failed with HTTP error: " +
                             httplib::to_string(res.error()) +
                             ". Make sure you're connected to the internet.");
  }

  if (res->status != 200) {
    throw std::runtime_error(
        "The licensing check failed with the response HTTP error code " +
        std::to_string(res->status));
  }

  json result_body = json::parse(res->body);

  bool is_valid = result_body["meta"]["valid"];
  std::string detail = result_body["meta"]["detail"];

  if (!is_valid) {
    throw std::runtime_error(
        "The licensing server says that your license is invalid. It returned "
        "the following message: " +
        detail);
  }

  verifySignature(res);
}
}  // namespace thirdai::licensing
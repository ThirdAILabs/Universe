#include <cryptopp/files.h>
#include <cryptopp/hex.h>
#include <cryptopp/integer.h>
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

  return signature_header.substr(signature_start,
                                 signature_end - signature_start);
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
  signature = get_signature(signature);

  // See https://www.cryptopp.com/wiki/SHA2
  std::string digest;
  CryptoPP::SHA256 hash;

  hash.Update(reinterpret_cast<const CryptoPP::byte*>(res->body.data()),
              res->body.length());
  digest.resize(hash.DigestSize());
  hash.Final(reinterpret_cast<CryptoPP::byte*>(&digest[0]));

  std::string encoded_digest;
  CryptoPP::StringSource( // NOLINT
      digest, true,
      new CryptoPP::Base64Encoder(                  
          new CryptoPP::StringSink(encoded_digest))  // Base64Encoder
  );

  if (!encoded_digest.empty()) {
    encoded_digest.pop_back();
  }

  std::stringstream signed_data_stream;
  signed_data_stream << "(request-target): post "
                        "/v1/accounts/thirdai/licenses/actions/validate-key\n"
                     << "host: api.keygen.sh\n"
                     << "date: " << res->get_header_value("date") << "\n"
                     << "digest: sha-256=" << encoded_digest;
  std::string signed_data = signed_data_stream.str();


  const CryptoPP::byte key[] = {0x9A, 0xDB, 0xFD, 0x88, 0x1D, 0x36, 0x3D, 0x31,
                                0xC1, 0x56, 0xC3, 0x48, 0x99, 0x68, 0x92, 0xDD,
                                0x01, 0x83, 0xA7, 0xE4, 0x54, 0x0D, 0xC0, 0x01,
                                0xCB, 0x7C, 0x2E, 0xEC, 0x5A, 0x79, 0x66, 0xAE};

  CryptoPP::Integer x(key, sizeof(key), CryptoPP::Integer::UNSIGNED, CryptoPP::LITTLE_ENDIAN_ORDER);

  CryptoPP::ed25519PublicKey pub_key;
  pub_key.SetPublicElement(x);

  CryptoPP::Integer public_element = pub_key.GetPublicElement();
  std::cout << public_element.ByteCount() << std::endl;

  CryptoPP::ed25519::Verifier verifier(pub_key);

  CryptoPP::Base64Encoder encoder(new CryptoPP::FileSink(std::cout));
  std::cout << "Public key: ";
  verifier.AccessPublicKey().Save(encoder);
  std::cout << std::endl;

  CryptoPP::Base64Encoder encoder2(new CryptoPP::FileSink(std::cout));
  std::cout << "Public key: ";
  verifier.AccessPublicKey().Save(encoder2);
  std::cout << std::endl;

  std::cout << signed_data << std::endl;
  std::cout << signed_data.size() << std::endl;
  std::cout << signature << std::endl;
  std::cout << signature.size() << std::endl;

  std::string decoded_signature;
   
  CryptoPP::StringSource ss(signature, true, // NOLINT
      new CryptoPP::Base64Decoder(
          new CryptoPP::StringSink(decoded_signature)
      ) // Base64Decoder
  ); // StringSource

  bool valid = verifier.VerifyMessage(
      reinterpret_cast<const CryptoPP::byte*>(signed_data.data()),
      signed_data.size(),
      reinterpret_cast<const CryptoPP::byte*>(decoded_signature.data()), decoded_signature.size());

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
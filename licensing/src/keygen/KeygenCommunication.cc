#define CPPHTTPLIB_OPENSSL_SUPPORT

#include "KeygenCommunication.h"
#include "cryptopp/sha.h"       // SHA256
#include "cryptopp/xed25519.h"  // Ed25519
#include "httplib.h"
#include "json.hpp"
#include <cryptopp/base64.h>  // Base64 decoder
#include <cryptopp/files.h>
#include <cryptopp/hex.h>
#include <cryptopp/integer.h>
#include <iostream>
#include <stdexcept>

namespace thirdai::licensing {

using json = nlohmann::json;

const std::string KEYGEN_PUBLIC_KEY_BASE64_DER =
    "MCowBQYDK2VwAyEAmtv9iB02PTHBVsNImWiS3QGDp+RUDcABy3wu7Fp5Zq4=";

/*
 * This method creates an ed25519Verifier that uses the thirdai Keygen public
 * key (KEYGEN_PUBLIC_KEY_BASE64_DER) for verifying messages signed with the
 * thirdai Keygen private key.
 */
CryptoPP::ed25519Verifier createVerifier() {
  CryptoPP::StringSource source(KEYGEN_PUBLIC_KEY_BASE64_DER, true,
                                new CryptoPP::Base64Decoder);
  return CryptoPP::ed25519::Verifier(source);
}

/*
 * This method returns a SHA256 hash on the input string.
 * See https://www.cryptopp.com/wiki/SHA2 for more details.
 */
std::string sha256(const std::string& input) {
  std::string digest;
  CryptoPP::SHA256 hash;

  hash.Update(reinterpret_cast<const CryptoPP::byte*>(input.data()),
              input.length());
  digest.resize(hash.DigestSize());
  hash.Final(reinterpret_cast<CryptoPP::byte*>(&digest[0]));

  return digest;
}

/*
 * This method returns the Base64 encoding of the passed in binary string.
 * See https://www.cryptopp.com/wiki/Base64Encoder for more details.
 */
std::string base64Encode(const std::string& input) {
  std::string encoded;
  CryptoPP::StringSource(  // NOLINT
      input, true,
      new CryptoPP::Base64Encoder(
          new CryptoPP::StringSink(encoded))  // Base64Encoder
  );

  // For some reason CryptoPP adds an extra new line at the end of the encoded
  // string so we need to pop it off here
  if (!encoded.empty()) {
    encoded.pop_back();
  }

  return encoded;
}

/*
 * Returns the exact message that was originally signed by Keygen. Assumes that
 * the passed in request was sent to the validate key Keygen endpoint (the
 * request target is part of the message signed by Keygen).
 */
std::string getOriginalKeygenMessage(const httplib::Result& res) {
  if (!res->has_header("date")) {
    throw std::runtime_error(
        "License was found to be valid, but did not find a date in the"
        "response (necessary to verify that the response came from Keygen).");
  }

  std::string body_hash = sha256(res->body);
  std::string base64_encoded_body_hash = base64Encode(body_hash);
  std::string date = res->get_header_value("date");

  std::stringstream original_keygen_message_stream;
  original_keygen_message_stream
      << "(request-target): post "
         "/v1/accounts/thirdai/licenses/actions/validate-key\n"
      << "host: api.keygen.sh\n"
      << "date: " << date << "\n"
      << "digest: sha-256=" << base64_encoded_body_hash;
  return original_keygen_message_stream.str();
}

/*
 * This method parses the signature from the request. The first step is getting
 * the Base64 representation of the signature, which we do by finding the
 * first occurance of signature="<key>" in the Keygen-Signature header and
 * parsing the key from that part of the string. We then convert it from Base64
 * into its true binary representation and return that.
 */
std::string getSignature(const httplib::Result& res) {
  if (!res->has_header("Keygen-Signature")) {
    throw std::runtime_error(
        "License was found to be valid, but did not find a Keygen signature "
        "verifying that the response came from Keygen.");
  }

  std::string signature_header = res->get_header_value("Keygen-Signature");

  // We add 11 because the actual signature starts 11 chars after the beginning
  // of the found string
  size_t signature_start = signature_header.find("signature=") + 11;
  // The signature ends one char before the next " after signature_start
  size_t signature_end = signature_header.find(',', signature_start) - 1;

  std::string base64_signature =
      signature_header.substr(signature_start, signature_end - signature_start);

  std::string decoded_signature;
  CryptoPP::StringSource ss(
      base64_signature, true,
      new CryptoPP::Base64Decoder(
          new CryptoPP::StringSink(decoded_signature))  // Base64Decoder
  );                                                    // StringSource

  return decoded_signature;
}

/*
 * See https://keygen.sh/docs/api/signatures/#response-signatures for how
 * Keygen expects us to verify responses. To summarize, we combine the returned
 * body with some other data in a very specific way, then apply the public
 * Ed25519 key from our Keygen account to verify the result is the same as the
 * signature (which we also parse from the header). The message we build and
 * verify is in the format
 *
 * (request-target): get /v1/accounts/thirdai/licenses/actions/validate-key\n
 * host: api.keygen.sh\n
 * date: Wed, 09 Jun 2021 16:08:15 GMT\n
 * digest: sha-256=827Op2un8OT9KJuN1siRs5h6mxjrUh4LJag66dQjnIM=
 *
 * where the digest is a sha-256 hash of the body of the request, converted into
 * a Base64 encoding.
 */
void verifyKeygenResponse(const httplib::Result& res) {
  std::string signed_data = getOriginalKeygenMessage(res);
  std::string signature = getSignature(res);
  CryptoPP::ed25519::Verifier verifier = createVerifier();

  bool valid = verifier.VerifyMessage(
      reinterpret_cast<const CryptoPP::byte*>(signed_data.data()),
      signed_data.size(),
      reinterpret_cast<const CryptoPP::byte*>(signature.data()),
      signature.size());

  if (!valid) {
    throw std::runtime_error(
        "We were not able to verify the response from the Keygen server.");
  }
}

/*
 * Communicates with the Keygen server to verify that the user with the given
 * access key is validated to use the ThirdAI python package.
 */
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

  verifyKeygenResponse(res);
}
}  // namespace thirdai::licensing
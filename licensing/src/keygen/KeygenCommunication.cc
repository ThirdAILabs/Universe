// This enables ssl support (which is required for https links), see
// https://github.com/yhirose/cpp-httplib for more details.
#include <string>
#include <unordered_set>
#define CPPHTTPLIB_OPENSSL_SUPPORT

#include "KeygenCommunication.h"
#include <cpp-httplib/httplib.h>
#include <cryptopp/base64.h>  // Base64 decoder
#include <cryptopp/files.h>
#include <cryptopp/integer.h>
#include <cryptopp/sha.h>       // SHA256
#include <cryptopp/xed25519.h>  // Ed25519
#include <json/include/nlohmann/json.hpp>
#include <licensing/src/utils.h>
#include <iostream>
#include <stdexcept>

namespace thirdai::licensing {

using json = nlohmann::json;

// DER describes the binary encoding format of the key, see
// https://www.cryptopp.com/wiki/Keys_and_Formats#BER_and_DER_Encoding
// for more details.
const std::string KEYGEN_PUBLIC_KEY_BASE64_DER =
    "MCowBQYDK2VwAyEAmtv9iB02PTHBVsNImWiS3QGDp+RUDcABy3wu7Fp5Zq4=";

const std::string VALIDATE_ENDPOINT =
    "/v1/accounts/thirdai/licenses/actions/validate-key";

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
 * the passed in HTTP result corresponds to a request that was sent to the
 * validate key Keygen endpoint, AKA:
 * "post /v1/accounts/thirdai/licenses/actions/validate-key"
 * This assumption is necessary because Keygen includes this data as part of
 * the message it signs.
 */
std::string getOriginalKeygenMessage(const httplib::Result& res,
                                     const std::string& request_type,
                                     const std::string& api_path) {
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
      << "(request-target): " << request_type << " " << api_path << "\n"
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
  // The signature ends two chars before the next , after signature_start, so
  // we subtract 1 from the index of the next , to get the exclusive end index
  size_t signature_end = signature_header.find(',', signature_start) - 1;

  std::string base64_signature =
      signature_header.substr(signature_start, signature_end - signature_start);

  std::string decoded_signature;
  CryptoPP::StringSource ss(
      /* string = */ base64_signature, /* pumpAll = */ true,
      /* attachment = */
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
 * (request-target): [get|post] api_path\n
 * host: api.keygen.sh\n
 * date: Wed, 09 Jun 2021 16:08:15 GMT\n
 * digest: sha-256=827Op2un8OT9KJuN1siRs5h6mxjrUh4LJag66dQjnIM=
 *
 * where the digest is a sha-256 hash of the body of the request, converted into
 * a Base64 encoding.
 */
void verifyKeygenResponse(const httplib::Result& res,
                          const std::string& request_type,
                          const std::string& api_path) {
  std::string signed_data =
      getOriginalKeygenMessage(res, request_type, api_path);
  std::string signature = getSignature(res);
  CryptoPP::ed25519::Verifier verifier =
      createVerifierFromBase64String(KEYGEN_PUBLIC_KEY_BASE64_DER);

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

void assertResponse200(const httplib::Result& response) {
  if (!response) {
    throw std::runtime_error("Licensing check failed with HTTP error: " +
                             httplib::to_string(response.error()) +
                             ". Make sure you're connected to the internet.");
  }

  if (response->status != 200) {
    throw std::runtime_error(
        "The licensing check failed with the response HTTP error code " +
        std::to_string(response->status) + " and body " + response->body);
  }
}

std::unordered_set<std::string> getKeygenEntitlements(
    const json& result_body, const std::string& access_key) {
  std::string user_entitlement_endpoint =
      result_body["data"]["relationships"]["entitlements"]["links"]["related"];

  // https://keygen.sh/docs/api/licenses/#licenses-relationships-list-entitlements
  httplib::Client client("https://api.keygen.sh");
  // We need this because for some strange reason building a wheel on github
  // actions and then installing locally makes ssl server certificate
  // verification fail. It isn't a huge problem to ignore it because we verify
  // all keygen resposes anyways.
  client.enable_server_certificate_verification(false);

  httplib::Headers headers = {{"Accept", "application/vnd.api+json"},
                              {"Authorization", "License " + access_key}};
  httplib::Result response = client.Get(
      /* path = */ user_entitlement_endpoint,
      /* headers = */ headers);
  assertResponse200(response);
  verifyKeygenResponse(response, /* request_type = */ "get",
                       /* api_path = */ user_entitlement_endpoint);

  json response_body = json::parse(response->body);
  std::unordered_set<std::string> result;
  for (const json& entitlement : response_body["data"]) {
    result.insert(entitlement["attributes"]["code"]);
  }
  return result;
}

std::unordered_set<std::string> verifyWithKeygen(
    const std::string& access_key) {
  httplib::Client client("https://api.keygen.sh");
  // We need this because for some strange reason building a wheel on github
  // actions and then installing locally makes ssl server certificate
  // verification fail. It isn't a huge problem to ignore it because we verify
  // all keygen resposes anyways.
  client.enable_server_certificate_verification(false);

  // These headers denote that we sending (Content-Type) and expect to receive
  // (Accept) json with additional "vendor specific" semantics. This is what
  // Keygen recommends, see
  // https://keygen.sh/docs/api/licenses/#licenses-actions-validate-key
  httplib::Headers headers = {{"Content-Type", "application/vnd.api+json"},
                              {"Accept", "application/vnd.api+json"}};
  json body;
  body["meta"]["key"] = access_key;

  httplib::Result response = client.Post(
      /* path = */ VALIDATE_ENDPOINT,
      /* headers = */ headers,
      /* body = */ body.dump(), /* content_type = */ "application/json");
  assertResponse200(response);

  json response_body = json::parse(response->body);
  bool license_is_valid = response_body["meta"]["valid"];
  std::string detail = response_body["meta"]["detail"];

  if (!license_is_valid) {
    throw std::runtime_error(
        "The licensing server says that your license is invalid. It returned "
        "the following message: " +
        detail);
  }

  verifyKeygenResponse(response, /* request_type = */ "post",
                       /* api_path = */ VALIDATE_ENDPOINT);

  return getKeygenEntitlements(response_body, access_key);
}
}  // namespace thirdai::licensing
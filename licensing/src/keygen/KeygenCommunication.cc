#include <stdexcept>
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "KeygenCommunication.h"
#include "httplib.h"
#include "json.hpp"
#include <iostream>

namespace thirdai::licensing {

using json = nlohmann::json;

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

  if (!res->has_header("Keygen-Signature")) {
    throw std::runtime_error(
        "License was found to be valid, but did not find a Keygen signature "
        "verifying that the response came from Keygen. ");
  }

  auto signature = res->get_header_value("Keygen-Signature");
  
}
}  // namespace thirdai::licensing
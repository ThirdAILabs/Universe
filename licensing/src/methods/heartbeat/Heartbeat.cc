#include "Heartbeat.h"
#include <exceptions/src/Exceptions.h>
#include <utils/UUID.h>
#include <stdexcept>
#include <string>
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <cpp-httplib/httplib.h>
#include <cryptopp/base64.h>  // Base64 decoder
#include <cryptopp/xed25519.h>
#include <json/include/nlohmann/json.hpp>
#include <licensing/src/Utils.h>
#include <chrono>

namespace thirdai::licensing {

using json = nlohmann::json;

const std::string THIRDAI_PUBLIC_KEY_BASE64_DER =
    "MCowBQYDK2VwAyEAqA9j+Pk81yUz7FPZfg94bez6m1j8j1jiLctTjmB2s7w=";

HeartbeatThread::HeartbeatThread(
    const std::string& url, const std::optional<uint32_t>& heartbeat_timeout)
    : _machine_id(utils::uuid::THIRDAI_UUID),
      _verified(true),
      _should_terminate(false) {
  if (!heartbeat_timeout.has_value()) {
    _no_heartbeat_grace_period_seconds = MAX_NO_HEARTBEAT_GRACE_PERIOD_SECONDS;
  } else {
    if (MAX_NO_HEARTBEAT_GRACE_PERIOD_SECONDS < *heartbeat_timeout) {
      throw std::invalid_argument(
          "Heartbeat timeout must be less than " +
          std::to_string(MAX_NO_HEARTBEAT_GRACE_PERIOD_SECONDS) + " seconds.");
    }
    _no_heartbeat_grace_period_seconds = *heartbeat_timeout;
  }

  if (!doSingleHeartbeat(url)) {
    throw exceptions::LicenseCheckException(
        "Could not establish initial connection to licensing server.");
  }
  _heartbeat_thread = std::thread(&HeartbeatThread::heartbeatThread, this, url);
}

void HeartbeatThread::verify() {
  if (!_verified) {
    throw exceptions::LicenseCheckException(
        "The heartbeat thread could not verify with the server because there "
        "has not been a successful heartbeat in " +
        std::to_string(_no_heartbeat_grace_period_seconds) +
        " seconds. Check the logs or metrics for more information.");
  }
}

HeartbeatThread::~HeartbeatThread() {
  _should_terminate = true;
  _heartbeat_thread.join();
}

int64_t currentEpochSeconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void HeartbeatThread::heartbeatThread(const std::string& url) {
  int64_t last_validation = currentEpochSeconds();
  while (!_should_terminate) {
    if (doSingleHeartbeat(url)) {
      last_validation = currentEpochSeconds();
      _verified = true;
    } else {
      _verified = currentEpochSeconds() - last_validation <
                  _no_heartbeat_grace_period_seconds;
    }

    std::this_thread::sleep_for(std::chrono::seconds(HEARTBEAT_PERIOD_SECONDS));
  }
}

bool verifyResponse(const std::string& submitted_machine_id,
                    const std::string& submitted_metadata,
                    const std::string& base64_signature) {
  const std::string& message = submitted_machine_id + "\n" + submitted_metadata;
  std::string decoded_signature;
  CryptoPP::StringSource ss(
      /* string = */ base64_signature, /* pumpAll = */ true,
      /* attachment = */
      new CryptoPP::Base64Decoder(
          new CryptoPP::StringSink(decoded_signature))  // Base64Decoder
  );
  CryptoPP::ed25519::Verifier verifier =
      createVerifierFromBase64String(THIRDAI_PUBLIC_KEY_BASE64_DER);
  return verifier.VerifyMessage(
      reinterpret_cast<const CryptoPP::byte*>(message.data()), message.size(),
      reinterpret_cast<const CryptoPP::byte*>(decoded_signature.data()),
      decoded_signature.size());
}

bool HeartbeatThread::doSingleHeartbeat(const std::string& url) {
  httplib::Client client(url);
  // See KeygenCommunication.cc for why we need this
  client.enable_server_certificate_verification(false);
  json body;
  body["machine_id"] = _machine_id;
  body["metadata"] = std::to_string(currentEpochSeconds());

  httplib::Result response = client.Post(
      /* path = */ "/heartbeat",
      /* headers = */ {},
      /* body = */ body.dump(), /* content_type = */ "application/json");
  if (!response || response->status != 200) {
    return false;
  }

  return verifyResponse(/* submitted_machine_id = */ body["machine_id"],
                        /* submitted_metadata = */ body["metadata"],
                        /* base64_signature = */ response->body);
}

}  // namespace thirdai::licensing
#include "Heartbeat.h"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <cpp-httplib/httplib.h>
#include <cryptopp/base64.h>  // Base64 decoder
#include <cryptopp/xed25519.h>
#include <json/include/nlohmann/json.hpp>
#include <chrono>

namespace thirdai::licensing {

using json = nlohmann::json;

const std::string THIRDAI_PUBLIC_KEY_BASE64_DER =
    "MCowBQYDK2VwAyEAqA9j+Pk81yUz7FPZfg94bez6m1j8j1jiLctTjmB2s7w=";

HeartbeatThread::HeartbeatThread(const std::string& url) {
  doSingleHeartbeat(url);
  _heartbeat_thread = std::thread(&HeartbeatThread::heartbeatThread, this, url);
}

void HeartbeatThread::verify() {
  if (!_verified) {
    throw std::runtime_error(
        "The heartbeat thread could not verify with the server, either "
        "because the initial heartbeat failed or there has not been a "
        "successful heartbeat in " +
        std::to_string(VALIDATION_FAIL_TIMEOUT_SECONDS) +
        " seconds. Check the logs or metrics for more information.");
  }
}

HeartbeatThread::~HeartbeatThread() {
  _should_terminate = true;
  _heartbeat_thread.join();
}

void HeartbeatThread::heartbeatThread(const std::string& url) {
  while (!_should_terminate) {
    std::this_thread::sleep_for(
        std::chrono::seconds(THREAD_SLEEP_PERIOD_SECONDS));
  }
  doSingleHeartbeat(url);
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

void HeartbeatThread::doSingleHeartbeat(const std::string& url) {
  httplib::Client client(url);
  // See KeygenCommunication.cc for why we need this
  client.enable_server_certificate_verification(false);
  client.Get("/heartbeat");
  json body;
  body["machine_id"] = _machine_id;
  body["metadata"] =
      std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count());

  httplib::Result response = client.Post(
      /* path = */ "/heartbeat",
      /* headers = */ {},
      /* body = */ body.dump(), /* content_type = */ "application/json");
  if (!response || response->status != 200) {
    return;
  }

  if (!verifyResponse(/* submitted_machine_id = */ body["machine_id"],
                      /* submitted_metadata = */ body["metadata"],
                      /* base64_signature = */ response->body)) {
    return;
  }

  _verified = true;
}

}  // namespace thirdai::licensing
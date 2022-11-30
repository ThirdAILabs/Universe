#include "KeygenCommunication.h"
#include <cpr/cpr.h>
// #include <utils/Logging.h>
#include <iostream>

namespace thirdai::licensing {

void KeygenCommunication::verifyWithKeygen(const std::string& access_key) {
//   const std::string body = fmt::format(R"body(
//         "meta": {{
//           "key": {}
//         }}
//     )body",
//                                        access_key);
  cpr::Response r = cpr::Post(
      cpr::Url{"https://api.keygen.sh/v1/accounts/thirdai/licenses/actions/"},
      cpr::Body{access_key});
  // try {
  //   http::Request request{
  //       "https://api.keygen.sh/v1/accounts/thirdai/licenses/actions/"
  //       "validate-key"};
  //   const std::string body = fmt::format(R"body(
  //         "meta": {{
  //           "key": {}
  //         }}
  //     )body",
  //                                        access_key);
  //   const auto response = request.send(
  //       /* method = */ "POST", /* body = */ body,
  //       /* headerFields = */
  //       {{"Content-Type", "application/vnd.api+json"},
  //        {"Accept", "application/vnd.api+json"}});

  //   std::cout << std::string{response.body.begin(), response.body.end()}
  //             << '\n';  // print the result
  // } catch (const std::exception& e) {
  //   std::cerr << "Request failed, error: " << e.what() << '\n';
  // }
}
}  // namespace thirdai::licensing
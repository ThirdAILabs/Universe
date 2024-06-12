#include "DenyPatterns.h"
#include <cctype>
#include <regex>
#include <string>
#include <unordered_map>

namespace thirdai::data::ner {

using ValidatorFn = std::function<bool(const std::string&)>;

const std::regex ID = std::regex(R"(\b[a-zA-Z0-9]{2,}\b)");

const std::regex VEHICLEUIN = std::regex(R"(\b[A-Z0-9]{3,20}\b)");

const std::regex AGE = std::regex(R"(\b\d{1,3}\b)");

const std::regex HEIGHT = std::regex(
    R"(\b(((\d{1,4}(\.\d{1,4})?)(\"|in|inches|\'|ft|feet|cm|centimeters|m|meters)?)|(\d(\'|ft|feet)\d{1,2}(\"|in|inches)))\b)");

const std::regex PIN = std::regex(R"(\b\d{3,6}\b)");

const std::regex AMOUNT = std::regex(R"(\b\d{1,6}(\.\d{1,2})?((?!\d)|\b))");

bool isId(const std::string& token) {
  std::smatch match;
  if (std::regex_search(token, match, ID)) {
    uint32_t digits = 0;
    for (const auto& c : match.str()) {
      if (std::isdigit(c)) {
        digits++;
      }
    }

    return (static_cast<float>(digits) / match.length()) >= 0.5;
  }
  return false;
}

const std::unordered_map<std::string, ValidatorFn> CHECKS = {
    {"UIN", isId},
    {"VEHICLEUIN",
     [](const std::string& token) {
       return std::regex_search(token, VEHICLEUIN);
     }},
    {"AGE",
     [](const std::string& token) { return std::regex_search(token, AGE); }},
    {"HEIGHT",
     [](const std::string& token) { return std::regex_search(token, HEIGHT); }},
    {"ACCOUNTNUMBER", isId},
    {"PIN",
     [](const std::string& token) { return std::regex_search(token, PIN); }},
    {
        "AMOUNT",
        [](const std::string& token) {
          return std::regex_search(token, AMOUNT);
        },
    },
};

bool allowed(const std::string& token, const std::string& entity) {
  if (CHECKS.count(entity)) {
    return CHECKS.at(entity)(token);
  }
  return true;
}

}  // namespace thirdai::data::ner
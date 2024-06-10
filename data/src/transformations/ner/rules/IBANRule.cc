#include <data/src/transformations/ner/rules/Pattern.h>
#include <utils/text/StringManipulation.h>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::data::ner {

enum BBANCharType {
  Alpha,
  Numeric,
  AlphaNumeric,
};

using BBANItem = std::pair<size_t, BBANCharType>;

BBANItem A(size_t len) { return {len, BBANCharType::Alpha}; }

BBANItem N(size_t len) { return {len, BBANCharType::Numeric}; }

BBANItem C(size_t len) { return {len, BBANCharType::AlphaNumeric}; }

using BBANFormat = std::vector<BBANItem>;

std::regex buildRegex(const std::string& country_code,
                      const BBANFormat& format) {
  if (!std::regex_match(country_code, std::regex("[A-Z]{2}"))) {
    throw std::invalid_argument(
        "Invalid IBAN country code. Must be 2 capital letters.");
  }

  std::string pattern = "(" + country_code + "|" + text::lower(country_code) +
                        R"()([0-9]{2}( )?))";

  for (const auto& [len, type] : format) {
    switch (type) {
      case BBANCharType::Alpha:
        pattern += R"(([a-zA-Z]( )?){)" + std::to_string(len) + "}";
        break;
      case BBANCharType::Numeric:
        pattern += R"(([0-9]( )?){)" + std::to_string(len) + "}";
        break;
      case BBANCharType::AlphaNumeric:
        pattern += R"(([a-zA-Z0-9]( )?){)" + std::to_string(len) + "}";
        break;
    }
  }

  return std::regex(pattern);
}

// This are from https://en.wikipedia.org/wiki/International_Bank_Account_Number
const std::vector<std::pair<std::string, BBANFormat>> COUNTRY_FORMAT_TABLE = {
    {"AL", {N(8), C(16)}},         // Albania
    {"AD", {N(8), C(12)}},         // Andorra
    {"AT", {N(16)}},               // Austria
    {"AZ", {A(4), C(20)}},         // Azerbaijan
    {"BH", {A(4), C(14)}},         // Bahrain
    {"BY", {C(4), N(4), C(16)}},   // Belarus
    {"BE", {N(12)}},               // Belgium
    {"BA", {N(16)}},               // Bosnia & Herzegovina
    {"BR", {N(23), A(1), C(1)}},   // Brazil
    {"BG", {A(4), N(6), C(8)}},    // Bulgaria
    {"CR", {N(18)}},               // Costa Rica
    {"HR", {N(17)}},               // Croatia
    {"CY", {N(8), C(16)}},         // Cyprus
    {"CZ", {N(20)}},               // Czech Repuplic
    {"DK", {N(14)}},               // Denmark
    {"DO", {C(4), N(20)}},         // Dominican Republic
    {"TL", {N(19)}},               // East Timor
    {"EG", {N(25)}},               // Egypt
    {"SV", {A(4), N(20)}},         // El Salvador
    {"EE", {N(16)}},               // Estonia
    {"FK", {A(2), N(12)}},         // Falkland Islands
    {"FO", {N(14)}},               // Faroe Islands
    {"FI", {N(14)}},               // Finland
    {"FR", {N(10), C(11), N(2)}},  // France
    {"GE", {A(2), N(16)}},         // Georigia
    {"DE", {N(18)}},               // Germany
    {"GI", {A(4), C(15)}},         // Gibraltar
    {"GR", {N(7), C(16)}},         // Greece
    {"GL", {N(14)}},               // Greenland
    {"GT", {C(4), C(20)}},         // Guatemala
    {"HU", {N(24)}},               // Hungary
    {"IS", {N(22)}},               // Iceland
    {"IQ", {A(4), N(15)}},         // Iraq
    {"IE", {A(4), N(6), N(8)}},    // Ireland
    {"IL", {N(19)}},               // Israel
    {"IT", {A(1), N(10), C(12)}},  // Italy
    {"JO", {A(4), N(4), C(18)}},   // Jordan
    {"KZ", {N(3), C(13)}},         // Kazakhstan
    {"XK", {N(4), N(10), N(2)}},   // Kosovo
    {"KW", {A(4), C(22)}},         // Kuwait
    {"LV", {A(4), C(13)}},         // Latvia
    {"LB", {N(4), C(20)}},         // Lebanon
    {"LY", {N(21)}},               // Libya
    {"LI", {N(5), C(12)}},         // Liechtenstein
    {"LT", {N(16)}},               // Lithuania
    {"LU", {N(3), C(13)}},         // Luxembourg
    {"MT", {A(4), N(5), C(18)}},   // Malta
    {"MR", {N(23)}},               // Mauritania
    {"MU", {A(4), N(19), A(3)}},   // Mauritius
    {"MC", {N(10), C(11), N(2)}},  // Monaco
    {"MD", {C(2), C(18)}},         // Moldova
    {"MN", {N(4), N(12)}},         // Mongolia
    {"ME", {N(18)}},               // Montenegro
    {"NL", {A(4), N(10)}},         // Netherlands
    {"MK", {N(3), C(10), N(2)}},   // North Macedonia
    {"NO", {N(11)}},               // Norway
    {"OM", {N(3), C(14)}},         // Oman
    {"PK", {A(4), C(16)}},         // Pakistan
    {"PS", {A(4), C(21)}},         // Palestinian Territories
    {"PL", {N(24)}},               // Poland
    {"PT", {N(21)}},               // Portugal
    {"QA", {A(4), C(21)}},         // Qatar
    {"RO", {A(4), C(16)}},         // Romania
    {"RU", {N(14), C(15)}},        // Russia
    {"LC", {A(4), C(24)}},         // Saint Lucia
    {"SM", {A(1), N(10), C(12)}},  // San Marino
    {"ST", {N(21)}},               // Sao Tome and Principe
    {"SA", {N(2), C(18)}},         // Saudi Arabia
    {"RS", {N(18)}},               // Serbia
    {"SC", {A(4), N(20), A(3)}},   // Seychelles
    {"SK", {N(20)}},               // Slovakia
    {"SI", {N(15)}},               // Slovenia
    {"SO", {N(4), N(3), N(12)}},   // Somalia
    {"ES", {N(20)}},               // Spain
    {"SD", {N(14)}},               // Sudan
    {"SE", {N(20)}},               // Sweden
    {"CH", {N(5), C(12)}},         // Switzerland
    {"TN", {N(20)}},               // Tunisia
    {"TR", {N(5), N(1), C(16)}},   // Turkey
    {"UA", {N(6), C(19)}},         // Ukraine
    {"AE", {N(3), N(16)}},         // United Arab Emirates
    {"GB", {A(4), N(14)}},         // United Kingdom
    {"VA", {N(3), N(15)}},         // Vatican City
    {"VG", {A(4), N(16)}},         // Virgin Islands British
};

std::unordered_map<std::string, std::regex> buildMap() {
  std::unordered_map<std::string, std::regex> map;

  for (const auto& [country_code, format] : COUNTRY_FORMAT_TABLE) {
    map[country_code] = buildRegex(country_code, format);
  }

  return map;
}

const std::unordered_map<std::string, std::regex> COUNTRY_FORMATS = buildMap();

bool ibanChecksum(const std::string& token) {
  std::string country_code = token.substr(0, 2);
  for (char& c : country_code) {
    c = std::toupper(c);
  }

  if (!COUNTRY_FORMATS.count(country_code)) {
    return false;
  }

  const auto& format = COUNTRY_FORMATS.at(country_code);

  /**
   * Because the main IBAN regex pattern matches on the range of lengths, it is
   * possible that if there are extra characters at the end of the IBAN number
   * they are still present here, this prunes them away leaving just the
   * relevant part of the value.
   */
  std::smatch match;
  if (!std::regex_search(token, match, format)) {
    return false;
  }

  std::string number = match.str();

  /**
   * Steps to Compute Checksum:
   *
   * 1. Move first for digits to the end.
   * 2. Replace each letter with a number. A,a -> 10, B,b -> 11, Z,z -> 35
   * 3. Check it equals 1 mod 97.
   * 4. To avoid overflow we can compute the mod incrementally because
   *    N * 10^K == (N mod P) * 10^K (mod P)
   *    This allows us to ensure that the intermediate value never grows too
   *    large.
   */
  uint64_t value = 0;
  for (char c : number.substr(4) + number.substr(0, 4)) {
    if (std::isdigit(c)) {
      value = value * 10 + (c - '0');
    } else if ('a' <= c && c <= 'z') {
      value = value * 100 + (c - 'a') + 10;
    } else if ('A' <= c && c <= 'Z') {
      value = value * 100 + (c - 'A') + 10;
    }
    if (value >= 1000000) {
      value = value % 97;
    }
  }

  return (value % 97) == 1;
}

RulePtr ibanRule() {
  size_t min_len = 100000000;
  size_t max_len = 0;
  for (const auto& [country, format] : COUNTRY_FORMAT_TABLE) {
    size_t len = 0;
    for (const auto& term : format) {
      len += term.first;
    }
    if (len > max_len) {
      max_len = len;
    }
    if (len < min_len) {
      min_len = len;
    }
  }

  return Pattern::make(
      /*entity=*/"IBAN",
      /*pattern=*/R"(\b[a-zA-Z]{2}([0-9]( )?){2}([a-zA-Z0-9]( )?){)" +
          std::to_string(min_len) + "," + std::to_string(max_len) + R"(}\b)",
      /*pattern_score=*/0.9,
      /*context_keywords=*/
      {
          {"iban", 0.1},
      },
      /*validator=*/ibanChecksum);
}

}  // namespace thirdai::data::ner
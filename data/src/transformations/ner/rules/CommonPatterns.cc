#include "CommonPatterns.h"
#include <data/src/transformations/ner/rules/Pattern.h>
#include <utils/text/StringManipulation.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::data::ner {

std::optional<ValidatorSubMatch> creditCardLuhnCheck(
    const std::string& number) {
  /*
   * Checks whether the number being passed satisifies the luhn's check. This is
   * useful for detecting credit card numbers.
   */
  std::string cleaned;
  for (char c : number) {
    if (std::isdigit(c)) {
      cleaned.push_back(c);
    }
  }

  if (cleaned.size() > 19 || cleaned.size() < 12) {
    return std::nullopt;
  }

  int sum = 0;
  bool alternate = false;
  for (auto it = cleaned.rbegin(); it != cleaned.rend(); ++it) {
    int n = (*it) - '0';
    if (alternate) {
      n *= 2;
      if (n > 9) {
        n -= 9;
      }
    }
    sum += n;
    alternate = !alternate;
  }

  if (sum % 10 == 0) {
    return {{0, number.size()}};
  }
  return std::nullopt;
}

std::optional<ValidatorSubMatch> medicalLicenseLuhnCheck(
    const std::string& number) {
  /*
   * Checks whether the number being passed satisifies the luhn's check. This is
   * useful for detecting credit card numbers.
   */
  std::string cleaned;
  for (char c : number) {
    if (std::isdigit(c)) {
      cleaned.push_back(c);
    }
  }
  if (cleaned.empty()) {
    return std::nullopt;
  }

  int sum = -(cleaned[cleaned.size() - 1] - '0');
  bool alternate = true;
  for (int i = cleaned.size() - 2; i >= 0; i--) {
    int n = cleaned[i] - '0';
    sum += alternate ? 2 * n : n;
    alternate = !alternate;
  }

  if (sum % 10 == 0) {
    return {{0, number.size()}};
  }
  return std::nullopt;
}

std::optional<ValidatorSubMatch> ipAddressValidator(const std::string& addr) {
  if (addr.find(':') != std::string::npos) {
    // ipv6
    auto double_colon = addr.find("::");
    if (double_colon != std::string::npos) {
      if (addr.find("::", double_colon + 2) != std::string::npos) {
        return std::nullopt;
      }
    } else {
      auto parts = text::split(addr, ':');
      if (parts.size() != 8) {
        return std::nullopt;
      }
    }
  } else {
    // ipv4
    auto parts = text::split(addr, '.');
    if (parts.size() != 4) {
      return std::nullopt;
    }

    for (auto part : parts) {
      if (part.size() > 1 && part[0] == '0') {
        return std::nullopt;
      }
      int value = std::stoi(part);
      if (value < 0 || value > 255) {
        return std::nullopt;
      }
    }
  }
  return ValidatorSubMatch(0, addr.size());
}

RulePtr creditCardPattern() {
  // https://baymard.com/checkout-usability/credit-card-patterns
  // https://en.wikipedia.org/wiki/Payment_card_number
  return Pattern::make(
      /*entity=*/"CREDITCARDNUMBER",
      /*pattern=*/
      R"(\b(([4613]\d{3})|(5[0-5]\d{2}))[- ]?(\d{3,4})[- ]?(\d{3,4})[- ]?(\d{3,5})\b)",
      /*pattern_score=*/0.8,
      /*context_keywords=*/
      {
          {"credit", 0.2},
          {"card", 0.2},
          {"visa", 0.2},
          {"mastercard", 0.2},
          {"discover", 0.2},
          {"amex", 0.2},
          {"debit", 0.2},
      },
      /*validator=*/creditCardLuhnCheck);
}

RulePtr emailPattern() {
  return Pattern::make(
      /*entity=*/"EMAIL",
      /*pattern=*/
      R"(\b((([!#$%&'*+\-/=?^_`{|}~\w])|([!#$%&'*+\-/=?^_`{|}~\w][!#$%&'*+\-/=?^_`{|}~\.\w]{0,}[!#$%&'*+\-/=?^_`{|}~\w]))[@]\w+([-.]\w+)*\.\w+([-.]\w+)*)\b)",
      /*pattern_score=*/0.6,
      /*context_keywords=*/
      {
          {"email", 0.4},
          {"gmail", 0.3},
          {"outlook", 0.1},
          {"contact", 0.1},
          {"mail", 0.1},
      });
}

RulePtr phonePattern() {
  return Pattern::make(
      /*entity=*/"PHONENUMBER",
      /*pattern=*/
      R"((([\+\b]\d{1,3}[\.\- \(]\(?)|\b|\()\d{3}\)?[\.\- ]?\d{3}[\.\- ]?\d{4}\b)",
      /*pattern_score=*/0.6,
      /*context_keywords=*/
      {
          {"phone", 0.2},
          {"cell", 0.2},
          {"mobile", 0.2},
          {"number", 0.2},
          {"tele", 0.2},
          {"call", 0.2},
          {"text", 0.1},
          {"contact", 0.1},
      });
}

RulePtr phoneWithoutAreaCodePattern() {
  /**
   * This rule is separate because it is really just a match on 7 consecutive
   * numbers, and so with only the pattern match the score is lower, but it can
   * still score highly with keyword matches in the context.
   */
  return Pattern::make(
      /*entity=*/"PHONENUMBER",
      /*pattern=*/
      R"(\b\d{3}[\.\- ]?\d{4}\b)",
      /*pattern_score=*/0.3,
      /*context_keywords=*/
      {
          {"phone", 0.4},
          {"cell", 0.4},
          {"mobile", 0.4},
          {"number", 0.4},
          {"tele", 0.4},
          {"telephone", 0.4},
          {"cellphone", 0.4},
          {"call", 0.3},
          {"text", 0.3},
          {"contact", 0.3},
      });
}

RulePtr medicalLicensePattern() {
  return Pattern::make(
      /*entity=*/"MEDICAL_LICENSE",
      /*pattern=*/
      R"(\b[abcdefghjklmprstuxABCDEFGHJKLMPRSTUX][a-zA-Z]\d{7}|[abcdefghjklmprstuxABCDEFGHJKLMPRSTUX]9\d{7}\b)",
      /*pattern_score=*/0.3,
      /*context_keywords=*/
      {
          {"medical", 0.5},
          {"license", 0.2},
          {"certificate", 0.3},
          {"dea", 0.4},
      },
      /*validator=*/medicalLicenseLuhnCheck);
}

RulePtr bankNumberPattern() {
  return Pattern::make(
      /*entity=*/"BANK_NUMBER",
      /*pattern=*/R"(\b[0-9]{8,17}\b)",
      /*pattern_score=*/0.3,
      /*context_keywords=*/
      {
          {"bank", 0.5},
          {"account", 0.2},
          {"check", 0.3},
          {"checking", 0.4},
          {"save", 0.3},
          {"saving", 0.4},
          {"debit", 0.3},
      });
}

RulePtr ssnPattern() {
  return Pattern::make(
      /*entity=*/"SSN",
      /*pattern=*/
      R"(\b([0-9]{3})([- ]?)([0-9]{2})([- ]?)([0-9]{4})\b)",
      /*pattern_score=*/0.4,
      /*context_keywords=*/
      {
          {"ssn", 0.6},
          {"social", 0.3},
          {"security", 0.2},
          {"ssid", 0.5},
      });
}

RulePtr cvvPattern() {
  return Pattern::make(
      /*entity=*/"CREDITCARDCVV",
      /*pattern=*/
      // TODO(Any): does \b make sense here or should it just be [^0-9] (i.e.
      // non digit)? For example CVV102
      R"(\b[0-9]{3}\b)",
      /*pattern_score=*/0.0,
      /*context_keywords=*/
      {
          {"cvv", 0.9},
          {"cvc", 0.9},
          {"cvn", 0.9},
          {"credit", 0.6},
          {"card", 0.6},
          {"visa", 0.6},
          {"mastercard", 0.6},
          {"discover", 0.6},
          {"amex", 0.6},
          {"debit", 0.6},
          {"security", 0.3},
          {"code", 0.3},
      });
}

RulePtr usDriversLicensePattern() {
  // https://ntsi.com/drivers-license-format/
  return Pattern::make(
      /*entity=*/"USDRIVERSLICENSE",
      /*pattern=*/
      R"(\b([A-Z][0-9]{3,6}|[A-Z][0-9]{5,9}|[A-Z][0-9]{6,8}|[A-Z][0-9]{4,8}|[A-Z][0-9]{9,11}|[A-Z]{1,2}[0-9]{5,6}|H[0-9]{8}|V[0-9]{6}|X[0-9]{8}|A-Z]{2}[0-9]{2,5}|[A-Z]{2}[0-9]{3,7}|[0-9]{2}[A-Z]{3}[0-9]{5,6}|[A-Z][0-9]{13,14}|[A-Z][0-9]{18}|[A-Z][0-9]{6}R|[A-Z][0-9]{9}|[A-Z][0-9]{1,12}|[0-9]{9}[A-Z]|[A-Z]{2}[0-9]{6}[A-Z]|[0-9]{8}[A-Z]{2}|[0-9]{3}[A-Z]{2}[0-9]{4}|[A-Z][0-9][A-Z][0-9][A-Z]|[0-9]{7,8}[A-Z])\b)",
      /*pattern_score=*/0.1,
      /*context_keywords=*/
      {
          {"drive", 0.5},
          {"license", 0.5},
          {"permit", 0.4},
          {"driving", 0.3},
          {"id", 0.4},
          {"identification", 0.4},
      });
}

RulePtr usPassportPattern() {
  return Pattern::make(
      /*entity=*/"USPASSPORT",
      /*pattern=*/R"(\b([A-Z][0-9]{8}|[0-9]{9})\b)",
      /*pattern_score*/ 0.1,
      /*context_keywords=*/
      {
          {"passport", 0.7},
          {"us", 0.3},
          {"travel", 0.3},
          {"united", 0.3},
          {"states", 0.3},
      });
}

RulePtr ipAddressPattern() {
  // https://techhub.hpe.com/eginfolib/networking/docs/switches/5120si/cg/5998-8491_l3-ip-svcs_cg/content/436042795.htm#:~:text=An%20IPv6%20address%20is%20divided,%3A09C0%3A876A%3A130B.
  return Pattern::make(
      /*entity=*/"IPADDRESS",
      /*pattern=*/
      R"(\b(\d{1,3}(\.\d{1,3}){3}\b)|((::)?[\da-fA-F]{1,4}(:?:[\da-fA-F]{1,4}){0,7}(::|\b)))",
      /*pattern_score=*/0.6,
      /*context_keywords=*/
      {
          {"ip", 0.3},
          {"address", 0.3},
          {"internet", 0.2},
      },
      /*validator=*/ipAddressValidator);
}

RulePtr getRuleForEntity(const std::string& entity) {
  if (entity == "CREDITCARDNUMBER") {
    return creditCardPattern();
  }
  if (entity == "EMAIL") {
    return emailPattern();
  }
  if (entity == "PHONENUMBER") {
    return phonePattern();
  }
  if (entity == "MEDICAL_LICENSE") {
    return medicalLicensePattern();
  }
  if (entity == "BANK_NUMBER") {
    return bankNumberPattern();
  }
  if (entity == "SSN") {
    return ssnPattern();
  }
  if (entity == "IBAN") {
    return ibanPattern();
  }
  if (entity == "CREDITCARDCVV") {
    return cvvPattern();
  }
  if (entity == "USDRIVERSLICENSE") {
    return usDriversLicensePattern();
  }
  if (entity == "USPASSPORT") {
    return usPassportPattern();
  }
  if (entity == "IPADDRESS") {
    return ipAddressPattern();
  }

  throw std::invalid_argument("No rule for entity '" + entity + "'.");
}

RulePtr getRuleForEntities(const std::vector<std::string>& entities) {
  std::vector<RulePtr> rules;
  rules.reserve(entities.size());

  for (const auto& entity : entities) {
    rules.push_back(getRuleForEntity(entity));
  }

  return RuleCollection::make(rules);
}

}  // namespace thirdai::data::ner

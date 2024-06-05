#include "CommonRules.h"
#include <data/src/transformations/ner/rules/Pattern.h>

namespace thirdai::data::ner {

bool creditCardLuhnCheck(const std::string& number) {
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
  return sum % 10 == 0;
}

bool medicalLicenseLuhnCheck(const std::string& number) {
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
    return false;
  }

  int sum = -(cleaned[cleaned.size() - 1] - '0');
  bool alternate = true;
  for (int i = cleaned.size() - 2; i >= 0; i--) {
    int n = cleaned[i] - '0';
    sum += alternate ? 2 * n : n;
    alternate = !alternate;
  }
  return sum % 10 == 0;
}

RulePtr creditCardRule() {
  return Pattern::make(
      /*entity=*/"CREDIT_CARD",
      /*pattern=*/
      R"((([4613]\d{3})|(5[0-5]\d{2}))[- ]?(\d{3,4})[- ]?(\d{3,4})[- ]?(\d{3,5}))",
      /*pattern_score=*/0.8,
      /*context_keywords=*/
      {
          {"credit", 0.2},
          {"card", 0.2},
          {"visa", 0.2},
          {"mastercard", 0.2},
          {"discover", 0.2},
          {"amex", 0.2},
      },
      /*validator=*/creditCardLuhnCheck);
}

RulePtr emailRule() {
  return Pattern::make(
      /*entity=*/"EMAIL",
      /*pattern=*/
      R"(((([!#$%&'*+\-/=?^_`{|}~\w])|([!#$%&'*+\-/=?^_`{|}~\w][!#$%&'*+\-/=?^_`{|}~\.\w]{0,}[!#$%&'*+\-/=?^_`{|}~\w]))[@]\w+([-.]\w+)*\.\w+([-.]\w+)*))",
      /*pattern_score=*/0.6,
      /*context_keywords=*/
      {
          {"email", 0.4},
          {"contact", 0.1},
          {"mail", 0.1},
      });
}

RulePtr phoneRule() {
  return Pattern::make(
      /*entity=*/"PHONE",
      /*pattern=*/
      R"((\+?\d+[\.\- ]?)?\(?\d{3}\)?[\.\- ]?\d{3}[\.\- ]?\d{4})",
      /*pattern_score=*/0.6,
      /*context_keywords=*/
      {
          {"phone", 0.2},
          {"cell", 0.2},
          {"mobile", 0.2},
          {"number", 0.2},
          {"tele", 0.2},
          {"telephone", 0.2},
          {"cellphone", 0.2},
          {"call", 0.2},
          {"text", 0.1},
          {"contact", 0.1},
      });
}

RulePtr medicalLicenseRule() {
  return Pattern::make(
      /*entity=*/"MEDICAL_LICENSE",
      /*pattern=*/
      R"([abcdefghjklmprstuxABCDEFGHJKLMPRSTUX][a-zA-Z]\d{7}|[abcdefghjklmprstuxABCDEFGHJKLMPRSTUX]9\d{7})",
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

RulePtr bankNumberRule() {
  return Pattern::make(
      /*entity=*/"BANK_NUMBER",
      /*pattern=*/R"([0-9]{8,17})",
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

RulePtr ssnRule() {
  return Pattern::make(
      /*entity=*/"SSN",
      /*pattern=*/
      R"(([0-9]{3})([- ]?)([0-9]{2})([- ]?)([0-9]{4}))",
      /*pattern_score=*/0.4,
      /*context_keywords=*/
      {
          {"ssn", 0.6},
          {"social", 0.3},
          {"security", 0.2},
          {"ssid", 0.5},
      });
}

RulePtr cvvRule() {
  return Pattern::make(
      /*entity=*/"CVV",
      /*pattern=*/
      R"([0-9]{3})",
      /*pattern_score=*/0.2,
      /*context_keywords=*/
      {
          {"cvv", 0.6},
          {"cvc", 0.6},
          {"card", 0.3},
      });
}

std::shared_ptr<Rule> defaultRule() {
  return RuleCollection::make({
      creditCardRule(),
      emailRule(),
      phoneRule(),
      medicalLicenseRule(),
      bankNumberRule(),
      ssnRule(),
      cvvRule(),
  });
}

}  // namespace thirdai::data::ner

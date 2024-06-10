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

RulePtr emailRule() {
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

RulePtr phoneRule() {
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
          {"telephone", 0.2},
          {"cellphone", 0.2},
          {"call", 0.2},
          {"text", 0.1},
          {"contact", 0.1},
      });
}

RulePtr phoneWithoutAreaCodeRule() {
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

RulePtr medicalLicenseRule() {
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

RulePtr bankNumberRule() {
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

RulePtr ssnRule() {
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

RulePtr cvvRule() {
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
      });
}

std::shared_ptr<Rule> defaultRule() {
  return RuleCollection::make({
      creditCardRule(),
      emailRule(),
      phoneRule(),
      phoneWithoutAreaCodeRule(),
      ibanRule(),
      // bankNumberRule(),
      // ssnRule(),
      cvvRule(),
  });
}

}  // namespace thirdai::data::ner

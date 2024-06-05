#include <gtest/gtest.h>
#include <data/src/transformations/ner/rules/CommonRules.h>
#include <data/src/transformations/ner/rules/Rule.h>
#include <string>

namespace thirdai::data::ner::tests {

TEST(NerRuleTests, TokenCleaning) {
  {
    std::vector<std::string> tokens{"Card", "924", "Number", "is",
                                    "240",  "249", "4",      "here"};
    std::vector<std::string> cleaned{"card",    "924",     "number",  "is",
                                     "2402494", "2402494", "2402494", "here"};

    ASSERT_EQ(Rule::cleanTokens(tokens), cleaned);
  }

  {
    std::vector<std::string> tokens{"Phone number", "(204)", "092-2490"};
    std::vector<std::string> cleaned{"phone number", "(204)092-2490",
                                     "(204)092-2490"};

    ASSERT_EQ(Rule::cleanTokens(tokens), cleaned);
  }

  {
    std::vector<std::string> tokens{"2024/24", "not", "nUMber"};
    std::vector<std::string> cleaned{"2024/24", "not", "number"};

    ASSERT_EQ(Rule::cleanTokens(tokens), cleaned);
  }

  {
    std::vector<std::string> tokens{"123", "456", "7-8.9", "10,20"};
    std::vector<std::string> cleaned{"1234567-8.9", "1234567-8.9",
                                     "1234567-8.9", "10,20"};

    ASSERT_EQ(Rule::cleanTokens(tokens), cleaned);
  }
}

void testRule(const RulePtr& rule, const std::vector<std::string>& tokens,
              size_t index, const std::string& entity, float score) {
  auto results = rule->apply(Rule::cleanTokens(tokens), index);

  if (score == 0) {
    ASSERT_TRUE(results.empty());
  } else {
    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].entity, entity);
    ASSERT_FLOAT_EQ(results[0].score, score);
  }
}

TEST(NerRuleTests, CreditCard) {
  auto rule = creditCardRule();

  testRule(rule, {"my", "card", "number", "is", "371449635398431", "and", "my"},
           4, "CREDIT_CARD", 1.0);

  testRule(rule,
           {"my", "card", "number", "is", "3714-4963-539-8431", "and", "my"}, 4,
           "CREDIT_CARD", 1.0);

  testRule(
      rule,
      {"my", "card", "number", "is", "3714", "4963", "539-8431", "and", "my"},
      5, "CREDIT_CARD", 1.0);

  testRule(rule,
           {"my", "number", "is", "3714", "4963", "539-8431", "and", "my"}, 5,
           "CREDIT_CARD", 0.8);

  testRule(rule,
           {"my", "number", "is", "3714", "4663", "539-8431", "and", "my"}, 5,
           "CREDIT_CARD", 0.0);

  testRule(rule, {"my", "card", "number", "is", "232", "and", "my"}, 4,
           "CREDIT_CARD", 0.0);
}

TEST(NerRuleTests, Email) {
  auto rule = emailRule();

  testRule(rule, {"my", "contacts", "joe@hotmail.co"}, 2, "EMAIL", 0.7);

  testRule(rule, {"joe@hotmail.co", "is", "my", "email"}, 0, "EMAIL", 1.0);

  testRule(rule, {"reach", "me", "at", "joe@hotmail.co", "during", "work"}, 3,
           "EMAIL", 0.6);

  testRule(rule, {"reach", "me", "at", "joe@hotmail", "during", "work"}, 3,
           "EMAIL", 0);

  testRule(rule, {"reach", "me", "at", "joe&hotmail.com", "during", "work"}, 3,
           "EMAIL", 0);
}

TEST(NerRuleTests, Phone) {
  auto rule = phoneRule();

  testRule(rule, {"for", "contacting", "(924)", "024-2400", "is", "my", "cell"},
           3, "PHONE", 0.9);

  testRule(rule, {"for", "reaching", "+1(924)", "024-2400", "is", "my", "cell"},
           3, "PHONE", 0.8);

  testRule(rule, {"9240242400"}, 0, "PHONE", 0.6);

  testRule(rule, {"940242400"}, 0, "PHONE", 0.0);

  testRule(rule, {"+1 9402412400"}, 0, "PHONE", 0.6);

  testRule(rule, {"+1 940.242-4200", "is", "my", "number"}, 0, "PHONE", 0.8);
}

TEST(NerRuleTests, MedicalLicense) {
  auto rule = medicalLicenseRule();

  testRule(rule, {"license", "no.", "BB1388568"}, 2, "MEDICAL_LICENSE", 0.5);

  testRule(rule, {"license", "no.", "BB1388558"}, 2, "MEDICAL_LICENSE", 0);

  testRule(rule, {"medical", "no.", "Ib1388568"}, 2, "MEDICAL_LICENSE", 0);
}

TEST(NerRuleTests, BankNumber) {
  auto rule = bankNumberRule();

  testRule(rule, {"my", "account", "is", "0123456789", "for", "savings"}, 3,
           "BANK_NUMBER", 0.9);

  testRule(rule, {"my", "info", "is", "0123456789", "for", "savings"}, 3,
           "BANK_NUMBER", 0.7);

  testRule(rule, {"my", "info", "is", "01", "for", "savings"}, 3, "BANK_NUMBER",
           0.0);
}

TEST(NerRuleTests, Ssn) {
  auto rule = ssnRule();

  testRule(rule, {"something", "123-24-0340", "something"}, 1, "SSN", 0.4);

  testRule(rule, {"ssn:", "something", "123", "24 2090", "something"}, 3, "SSN",
           1.0);

  testRule(rule, {"something", "123456789", "something", "social"}, 1, "SSN",
           0.7);

  testRule(rule, {"ssn:", "something", "1234", "something"}, 2, "SSN", 0);

  testRule(rule, {"security", "something", "123?42-2013", "something"}, 2,
           "SSN", 0);
}

TEST(NerRuleTests, Cvv) {
  auto rule = cvvRule();

  testRule(rule, {"something", "123", "something"}, 1, "CVV", 0.2);

  testRule(rule, {"cvv:", "something", "123", "something"}, 2, "CVV", 0.8);

  testRule(rule, {"something", "123", "cvc:", "something"}, 1, "CVV", 0.8);

  testRule(rule, {"cvc:", "something", "1234", "something"}, 2, "CVV", 0);
}

}  // namespace thirdai::data::ner::tests
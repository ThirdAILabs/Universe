#include <gtest/gtest.h>
#include <data/src/transformations/ner/rules/CommonPatterns.h>
#include <data/src/transformations/ner/rules/DenyPatterns.h>
#include <data/src/transformations/ner/rules/Rule.h>
#include <utils/text/StringManipulation.h>
#include <string>

namespace thirdai::data::ner::tests {

void checkNoMatch(const RulePtr& rule, const std::string& phrase) {
  auto results = rule->apply(phrase);

  ASSERT_TRUE(results.empty());
}

void checkMatch(const RulePtr& rule, const std::string& phrase,
                const std::string& entity, float score,
                const std::string& match) {
  auto results = rule->apply(phrase);

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].entity, entity);
  ASSERT_FLOAT_EQ(results[0].score, score);
  ASSERT_EQ(phrase.substr(results[0].start, results[0].len), match);
}

TEST(NerRuleTests, CreditCard) {
  auto rule = creditCardPattern();

  checkMatch(rule, "my card number is 371449635398431 and something else",
             "CREDITCARDNUMBER", 1.0, "371449635398431");

  checkMatch(rule, "my card number is 3714-4963-539-8431 and my",
             "CREDITCARDNUMBER", 1.0, "3714-4963-539-8431");

  checkMatch(rule, "my card number is 3714 4963 539-8431 and my",
             "CREDITCARDNUMBER", 1.0, "3714 4963 539-8431");

  checkMatch(rule, "my number is 3714 4963 539 8431 and my", "CREDITCARDNUMBER",
             0.8, "3714 4963 539 8431");

  checkNoMatch(rule, "my number is 3714 4663 539-8431 and my");

  checkNoMatch(rule, "my card number is 232 and my");
}

TEST(NerRuleTests, Email) {
  auto rule = emailPattern();

  checkMatch(rule, "my contact is joe@address.co", "EMAIL", 0.7,
             "joe@address.co");

  checkMatch(rule, "alex@address.co is my email", "EMAIL", 1.0,
             "alex@address.co");

  checkMatch(rule, "reach me at jane@address.co during work", "EMAIL", 0.6,
             "jane@address.co");

  checkMatch(rule, "reach me at joe@gmail.com during work", "EMAIL", 1.0,
             "joe@gmail.com");

  checkNoMatch(rule, "reach me at joe@gmail during work");

  checkNoMatch(rule, "reach me at joe&gmail.com during work");
}

TEST(NerRuleTests, Phone) {
  auto rule = phonePattern();

  checkMatch(rule, "for contacting (924) 024-2400 is my cell", "PHONENUMBER",
             0.9, "(924) 024-2400");

  checkMatch(rule, "for reaching +1 (924) 024-2400 is my cell", "PHONENUMBER",
             0.8, "+1 (924) 024-2400");

  checkMatch(rule, "for reaching +1(924) 024-2400 is my cell", "PHONENUMBER",
             0.8, "+1(924) 024-2400");

  checkMatch(rule, "9240242400", "PHONENUMBER", 0.6, "9240242400");

  checkNoMatch(rule, "940242400");

  checkMatch(rule, "something something +1 9402412400 something", "PHONENUMBER",
             0.6, "+1 9402412400");

  checkMatch(rule, "+1 940.242-4200 is my number", "PHONENUMBER", 0.8,
             "+1 940.242-4200");
}

TEST(NerRuleTests, MedicalLicense) {
  auto rule = medicalLicensePattern();

  checkMatch(rule, "license no. BB1388568", "MEDICAL_LICENSE", 0.5,
             "BB1388568");

  checkNoMatch(rule, "license no. BB1388558");

  checkNoMatch(rule, "medical no. Ib1388568");
}

TEST(NerRuleTests, BankNumber) {
  auto rule = bankNumberPattern();

  checkMatch(rule, "my account is 0123456789 for savings", "BANK_NUMBER", 0.9,
             "0123456789");

  checkMatch(rule, "my info is 0123456789 for savings", "BANK_NUMBER", 0.7,
             "0123456789");

  checkNoMatch(rule, "my info is 01 for savings");
}

TEST(NerRuleTests, Ssn) {
  auto rule = ssnPattern();

  checkMatch(rule, "something 123-24-0340 something", "SSN", 0.4,
             "123-24-0340");

  checkMatch(rule, "ssn: something 123 24 2090 something", "SSN", 1.0,
             "123 24 2090");

  checkMatch(rule, "something 123456789 something social", "SSN", 0.7,
             "123456789");

  checkNoMatch(rule, "ssn: something 1234 something");

  checkNoMatch(rule, "security something 123?42-2013 something");
}

TEST(NerRuleTests, Cvv) {
  auto rule = cvvPattern();

  checkMatch(rule, "cvv: something 123 something", "CREDITCARDCVV", 0.9, "123");

  checkMatch(rule, "something 123 cvc something", "CREDITCARDCVV", 0.9, "123");

  checkMatch(rule, "card is ... and 123 something", "CREDITCARDCVV", 0.6,
             "123");

  // 3 digit numbers on their own shouldn't match
  checkNoMatch(rule, "something 123 something");

  checkNoMatch(rule, "cvc: something 1234 something");
}

TEST(NerRuleTests, Iban) {
  auto rule = ibanPattern();

  checkMatch(rule, "my iban is DE89 3704 0044 0532 0130 00.", "IBAN", 1.0,
             "DE89 3704 0044 0532 0130 00");

  checkMatch(rule, "my num is DE89 3704 0044 0532 0130 00,more text", "IBAN",
             0.9, "DE89 3704 0044 0532 0130 00");

  checkMatch(rule, "my num is DE89 3704 0044 0532 0130 00abc", "IBAN", 0.9,
             "DE89 3704 0044 0532 0130 00");

  checkNoMatch(rule, "my iban is DE82 3704 0044 0532 0130 00.");
}

TEST(NerRuleTests, MultipleRules) {
  RulePtr rule = RuleCollection::make({
      creditCardPattern(),
      emailPattern(),
      phonePattern(),
      bankNumberPattern(),
      ssnPattern(),
      cvvPattern(),
  });

  std::vector<std::string> tokens = {
      "my",           "ssn",    "is",         "123",         "24",
      "0340",         "and",    "9249242001", "is",          "my",
      "phone",        "number", "and",        "credit:3714", "4963",
      "5398-431:end", "cvv",    "123",        "text"};

  auto results = rule->apply(tokens);

  auto verify_tags = [&tokens](const std::vector<TagsAndScores>& tags) {
    std::vector<std::vector<std::string>> entities = {
        {},
        {},
        {},
        {"SSN"},
        {"SSN"},
        {"SSN"},
        {},
        {"PHONENUMBER", "BANK_NUMBER"},
        {},
        {},
        {},
        {},
        {},
        {"CREDITCARDNUMBER"},
        {"CREDITCARDNUMBER"},
        {"CREDITCARDNUMBER", "CREDITCARDCVV"},
        {},
        {"CREDITCARDCVV"},
        {}};

    ASSERT_EQ(tags.size(), entities.size());

    for (size_t i = 0; i < tags.size(); i++) {
      ASSERT_EQ(tags.at(i).size(), entities.at(i).size());

      for (size_t j = 0; j < tags.at(i).size(); j++) {
        ASSERT_EQ(tags.at(i).at(j).first, entities.at(i).at(j));
      }
    }
  };

  verify_tags(results);

  auto batch_results = rule->applyBatch({tokens, tokens, tokens});

  for (const auto& res : batch_results) {
    verify_tags(res);
  }
}

TEST(NerRuleTests, DenyPatterns) {
  ASSERT_TRUE(allowed("A02409", "UIN"));
  ASSERT_TRUE(allowed("8247", "UIN"));
  ASSERT_FALSE(allowed("ALKFJB", "UIN"));

  ASSERT_TRUE(allowed("Z249KLF", "VEHICLEUIN"));
  ASSERT_FALSE(allowed("Z?LF", "VEHICLEUIN"));
  ASSERT_FALSE(allowed("banana", "VEHICLEUIN"));

  ASSERT_TRUE(allowed("19", "AGE"));
  ASSERT_FALSE(allowed("years", "AGE"));

  ASSERT_TRUE(allowed("4'9\"", "HEIGHT"));
  ASSERT_TRUE(allowed("4ft9in", "HEIGHT"));
  ASSERT_TRUE(allowed("19ft", "HEIGHT"));
  ASSERT_TRUE(allowed("29in", "HEIGHT"));
  ASSERT_TRUE(allowed("19'", "HEIGHT"));
  ASSERT_TRUE(allowed("29\"", "HEIGHT"));
  ASSERT_TRUE(allowed("80cm", "HEIGHT"));
  ASSERT_TRUE(allowed("1.409m", "HEIGHT"));
  ASSERT_TRUE(allowed("24.209", "HEIGHT"));
  ASSERT_TRUE(allowed("24.209feet", "HEIGHT"));
  ASSERT_FALSE(allowed("apple", "HEIGHT"));

  ASSERT_TRUE(allowed("A02409", "ACCOUNTNUMBER"));
  ASSERT_TRUE(allowed("8247", "ACCOUNTNUMBER"));
  ASSERT_FALSE(allowed("ALKFJB", "ACCOUNTNUMBER"));

  ASSERT_TRUE(allowed("2049", "PIN"));
  ASSERT_FALSE(allowed("2AZ9", "PIN"));

  ASSERT_TRUE(allowed("$24.02", "AMOUNT"));
  ASSERT_TRUE(allowed("$24k", "AMOUNT"));
  ASSERT_TRUE(allowed("£24", "AMOUNT"));
  ASSERT_TRUE(allowed("9824.42", "AMOUNT"));
  ASSERT_FALSE(allowed("apple", "AMOUNT"));
}

}  // namespace thirdai::data::ner::tests
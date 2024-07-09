#pragma once

#include <data/src/transformations/ner/rules/Rule.h>
#include <unordered_set>

namespace thirdai::data::ner {

RulePtr creditCardPattern();

RulePtr emailPattern();

RulePtr phonePattern();

RulePtr medicalLicensePattern();

RulePtr bankNumberPattern();

RulePtr ibanPattern();

RulePtr ssnPattern();

RulePtr cvvPattern();

RulePtr usDriversLicensePattern();

RulePtr usPassportPattern();

RulePtr ipAddressPattern();

RulePtr getRuleForEntity(const std::string& entity);

RulePtr getRuleForEntities(const std::vector<std::string>& entities);

inline std::unordered_set<std::string> common_entities = {
    "CREDITCARDNUMBER", "EMAIL", "MEDICAL_LICENSE",
    "BANK_NUMBER",      "SSN",   "IBAN",
    "CREDITCARDCVV"};
}  // namespace thirdai::data::ner

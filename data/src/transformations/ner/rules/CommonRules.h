#pragma once

#include <data/src/transformations/ner/rules/Rule.h>

namespace thirdai::data::ner {

RulePtr creditCardRule();

RulePtr emailRule();

RulePtr phoneRule();

RulePtr medicalLicenseRule();

RulePtr bankNumberRule();

RulePtr ibanRule();

RulePtr ssnRule();

RulePtr cvvRule();

RulePtr getRuleForEntity(const std::string& entity);

RulePtr getRuleForEntities(const std::vector<std::string>& entities);

}  // namespace thirdai::data::ner

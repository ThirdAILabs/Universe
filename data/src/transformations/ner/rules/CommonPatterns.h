#pragma once

#include <data/src/transformations/ner/rules/Rule.h>

namespace thirdai::data::ner {

RulePtr creditCardPattern();

RulePtr emailPattern();

RulePtr phonePattern();

RulePtr medicalLicensePattern();

RulePtr bankNumberPattern();

RulePtr ibanPattern();

RulePtr ssnPattern();

RulePtr cvvPattern();

RulePtr getRuleForEntity(const std::string& entity);

RulePtr getRuleForEntities(const std::vector<std::string>& entities);

}  // namespace thirdai::data::ner

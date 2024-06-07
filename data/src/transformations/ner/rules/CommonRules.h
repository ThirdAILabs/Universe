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

RulePtr defaultRule();

}  // namespace thirdai::data::ner

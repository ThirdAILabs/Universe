#pragma once

#include <string>

namespace thirdai::data::ner {

bool allowed(const std::string& token, const std::string& entity);

}  // namespace thirdai::data::ner
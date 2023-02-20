#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset::parsers::CSV {

std::vector<std::string_view> parseLine(const std::string& line,
                                        char delimiter);

}  // namespace thirdai::dataset::parsers::CSV
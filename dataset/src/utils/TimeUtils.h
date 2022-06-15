#pragma once

#include <ctime>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace thirdai::dataset {
class TimeUtils {
 public:
  static std::tm timeStringToTimeObject(std::string_view time_string) {
    std::tm time = {};
    std::stringstream time_ss;
    time_ss << time_string;

    if (time_ss >> std::get_time(&time, "%Y-%m-%d")) {
      return time;
    }

    std::stringstream error_ss;
    error_ss
        << "[TimeUtils::timeStringToTimeObject] Failed to parse the string '"
        << time_string
        << "'. Expected a timestamp string in the 'YYYY-MM-DD' format.";

    throw std::invalid_argument(error_ss.str());
  }
};
}  // namespace thirdai::dataset
#pragma once

#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

namespace thirdai::dataset {
class TimeUtils {
 public:
  // We use 64-bit int for timestamps
  static constexpr int64_t SECONDS_IN_DAY = static_cast<int64_t>(60 * 60 * 24);
  static constexpr int64_t SECONDS_IN_HOUR = static_cast<int64_t>(60 * 60);

  /**
   * Theres an STL function that does this (std::mktime)
   * but it is prohibitively slow as it reads a file from
   * disk whenever it is called. This also makes the STL
   * function difficult to parallelize. So we came up with
   * our own converter.
   *
   * Adapted from
   * https://gmbabar.wordpress.com/2010/12/01/mktime-slow-use-custom-function/
   */
  static int64_t timeToEpoch(const struct tm& time) {
    int64_t days_since_1970 = daysFrom1970ToYear(time) + dayOfYear(time);
    return days_since_1970 * SECONDS_IN_DAY;
  }

  // This is a convenience function.
  static std::exception_ptr timeStringToTimeObject(std::string_view time_string,
                                                   std::tm& time_object) {
    std::stringstream time_ss;
    time_ss << time_string;

    if (time_ss >> std::get_time(&time_object, "%Y-%m-%d")) {
      return nullptr;
    }

    std::stringstream error_ss;
    error_ss
        << "[TimeUtils::timeStringToTimeObject] Failed to parse the string '"
        << time_string
        << "'. Expected a timestamp string in the 'YYYY-MM-DD' format.";

    return std::make_exception_ptr(std::invalid_argument(error_ss.str()));
  }

 private:
  static constexpr int daysFrom1970ToYear(const struct tm& time) {
    int years_since_1970 = time.tm_year - 70;  // tm->tm_year is from 1900.
    int leaps_since_1970_to_year = (years_since_1970 + 1) / 4;
    return years_since_1970 * 365 + leaps_since_1970_to_year;
  }

  static constexpr int dayOfYear(const struct tm& time) {
    const int mon_days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int day_of_year = 0;

    for (int i = 0; i < time.tm_mon; i++) {
      day_of_year += mon_days[i];
    }
    if (isLeapYear(time) && pastFebruary(time)) {
      day_of_year++;
    }
    day_of_year += time.tm_mday - 1;  // Day of month is 1-indexed.

    return day_of_year;
  }

  static constexpr bool isLeapYear(const struct tm& time) {
    return (time.tm_year & 3) == 0;  // 1) tm_year is years since 1900.
                                     // 2) "& 3" is the same as "% 4".
  }

  static constexpr bool pastFebruary(const struct tm& time) {
    return time.tm_mon >= 2;  // Months are 0-indexed.
  }
};

}  // namespace thirdai::dataset
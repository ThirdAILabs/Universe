#pragma once

#include <cmath>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

namespace thirdai::dataset {

class TimeObject {
 public:
  // We use 64-bit int for timestamps
  static constexpr int64_t SECONDS_IN_DAY = static_cast<int64_t>(60 * 60 * 24);
  static constexpr int64_t SECONDS_IN_HOUR = static_cast<int64_t>(60 * 60);

  TimeObject() : _time_object() {}

  explicit TimeObject(const std::string_view& time_string) : _time_object() {
    std::stringstream time_ss;
    time_ss << time_string;

    if (time_ss >> std::get_time(&_time_object, "%Y-%m-%d")) {
      return;
    }

    std::stringstream error_ss;
    error_ss << "[Time] Failed to parse the string '" << time_string
             << "'. Expected a timestamp string in the 'YYYY-MM-DD' format.";

    throw std::invalid_argument(error_ss.str());
  }

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
  int64_t secondsSinceEpoch() const {
    int64_t days_since_1970 = daysFrom1970ToYear() + dayOfYear();
    return days_since_1970 * SECONDS_IN_DAY;
  }

  inline int month() const { return _time_object.tm_mon; }

  inline int dayOfMonthZeroIndexed() const { return _time_object.tm_mday - 1; }

  int dayOfYear() const {
    const int days_before_month[] = {0,   31,  59,  90,  120, 151,
                                     181, 212, 243, 273, 304, 334};
    int day_of_year = days_before_month[_time_object.tm_mon];

    if (isLeapYear() && pastFebruary()) {
      day_of_year++;
    }

    day_of_year += dayOfMonthZeroIndexed();

    return day_of_year;
  }

 private:
  int daysFrom1970ToYear() const {
    int years_since_1970 = yearsSince1900() - 70;
    return years_since_1970 * 365 +
           leapsBetweenJan1970AndYear(years_since_1970);
  }

  static int leapsBetweenJan1970AndYear(int years_since_1970) {
    return std::floor(static_cast<float>(years_since_1970 + 1) /
                      4);  // Just trust me on this :)
  }

  inline bool isLeapYear() const {
    return (yearsSince1900() & 3) == 0;  // "& 3" is the same as "% 4".
  }

  inline bool pastFebruary() const {
    int february_0_indexed = 1;
    return _time_object.tm_mon > february_0_indexed;
  }

  inline int yearsSince1900() const { return _time_object.tm_year; }

  struct tm _time_object;
};

}  // namespace thirdai::dataset
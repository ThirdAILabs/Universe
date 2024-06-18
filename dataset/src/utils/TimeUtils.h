#pragma once

#include <cmath>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

class TimeObject {
 public:
  // We use 64-bit int for timestamps
  static constexpr int64_t SECONDS_IN_DAY = static_cast<int64_t>(60 * 60 * 24);
  static constexpr int64_t SECONDS_IN_HOUR = static_cast<int64_t>(60 * 60);

  TimeObject() : _time_object() {}

  explicit TimeObject(const std::string_view& time_string,
                      const std::string& format = "%Y-%m-%d")
      : _time_object() {
    std::stringstream time_ss;
    time_ss << time_string;

    if (time_ss >> std::get_time(&_time_object, format.data())) {
      return;
    }

    std::stringstream error_ss;
    error_ss << "Failed to parse the string '" << time_string
             << "'. Expected a timestamp string in the '" << format
             << "' format.";

    throw std::invalid_argument(error_ss.str());
  }

  explicit TimeObject(const time_t seconds_since_epoch) {
    // Explicit type for windows compiler
    const time_t* const ptr = &seconds_since_epoch;
    _time_object = *std::gmtime(ptr);
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

  std::string string() const {
    std::stringstream ss;
    ss << 1900 + _time_object.tm_year << '-';
    if (_time_object.tm_mon < 9) {
      ss << '0';
    }
    ss << _time_object.tm_mon + 1 << '-';
    if (_time_object.tm_mday < 10) {
      ss << '0';
    }
    ss << _time_object.tm_mday;
    return ss.str();
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
    // "& 3" is the same as "% 4" but faster.
    return (yearsSince1900() & 3) == 0;
  }

  inline bool pastFebruary() const {
    int february_0_indexed = 1;
    return _time_object.tm_mon > february_0_indexed;
  }

  inline int yearsSince1900() const { return _time_object.tm_year; }

  struct tm _time_object;
};

inline std::string getDayOfWeek(uint32_t day_number) {
  if (day_number >= 7) {
    throw std::invalid_argument(
        "Day number must be between 0 and 6. Received " +
        std::to_string(day_number) + ".");
  }

  std::vector<std::string> day_names = {"Sunday",    "Monday",   "Tuesday",
                                        "Wednesday", "Thursday", "Friday",
                                        "Saturday"};

  return day_names[day_number];
}

inline std::string getMonthOfYear(uint32_t month_number) {
  if (month_number >= 12) {
    throw std::invalid_argument(
        "Month number must be between 0 and 11. Received " +
        std::to_string(month_number) + ".");
  }

  std::vector<std::string> month_names = {
      "January", "February", "March",     "April",   "May",      "June",
      "July",    "August",   "September", "October", "November", "December"};

  return month_names[month_number];
}

}  // namespace thirdai::dataset
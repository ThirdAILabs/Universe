#pragma once

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

namespace thirdai::dataset {
class TimeUtils {
 public:
  static constexpr uint32_t SECONDS_IN_DAY = 60 * 60 * 24;

  static time_t timeToEpoch(const struct tm* ltm, int utcdiff) {
    const int mon_days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int32_t tyears, tdays, leaps, utc_hrs;
    int tyears_for_leaps, i;

    tyears = ltm->tm_year - 70;  // tm->tm_year is from 1900.
    tyears_for_leaps = tyears;
    if (ltm->tm_mon < 2) {
      tyears_for_leaps--;
    }
    leaps =
        (tyears_for_leaps + 2) / 4;  // no of next two lines until year 2100.
    tdays = 0;
    for (i = 0; i < ltm->tm_mon; i++) {
      tdays += mon_days[i];
    }

    tdays += ltm->tm_mday - 1;  // days of month passed.
    tdays = tdays + (tyears * 365) + leaps;

    utc_hrs = ltm->tm_hour + utcdiff;  // for your time zone.
    return (tdays * 86400) + (utc_hrs * 3600);
  }

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

class TimestampGenerator {
 public:
  explicit TimestampGenerator(time_t start_timestamp)
      : _cur_timestamp(start_timestamp) {}

  explicit TimestampGenerator(const std::string& start_timestamp)
      : _cur_timestamp(0) {
    auto tm = dataset::TimeUtils::timeStringToTimeObject(start_timestamp);
    _cur_timestamp = dataset::TimeUtils::timeToEpoch(&tm, /* utcdiff = */ 0);
  }

  void addDays(time_t days) {
    _cur_timestamp += days * dataset::TimeUtils::SECONDS_IN_DAY;
  }

  std::string currentTimeString() {
    auto* tm = std::localtime(&_cur_timestamp);
    std::stringstream ss;
    ss << (1900 + tm->tm_year) << "-";
    if (tm->tm_mon < 9) {
      ss << "0";
    }
    ss << (tm->tm_mon + 1) << "-";
    if (tm->tm_mday < 10) {
      ss << "0";
    }
    ss << tm->tm_mday;
    return ss.str();
  }

 private:
  time_t _cur_timestamp;
};
}  // namespace thirdai::dataset
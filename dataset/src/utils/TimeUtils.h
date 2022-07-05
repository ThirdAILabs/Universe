#pragma once

#include <ctime>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace thirdai::dataset {
class TimeUtils {
 public:
  static time_t timeToEpoch(const struct tm* ltm, int utcdiff) {
    const int mon_days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int32_t tyears, tyears_for_leaps, tdays, leaps, utc_hrs;
    int i;

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

    if (time_ss >>
        std::get_time(&time, "%Y-%m-%d")) {  // TODO(Geordie): This uses local
                                             // time... is this bad?
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
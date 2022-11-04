#pragma once

#include <cereal/access.hpp>
#include <utils/StringManipulation.h>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace thirdai::utils {

class Duration {
 public:
  Duration(uint32_t n, std::string unit)
      : _in_seconds(n * parseUnit(std::move(unit))) {}

  uint32_t inSeconds() const { return _in_seconds; }

 private:
  static constexpr uint32_t SECONDS_IN_DAY = 86400;

  static uint32_t parseUnit(std::string unit) {
    unit = utils::lower(unit);

    if (unit == "s" || unit == "second") {
      return 1;
    }

    if (unit == "m" || unit == "minute") {
      return 60;
    }

    if (unit == "h" || unit == "hour") {
      return 3600;
    }

    if (unit == "d" || unit == "day") {
      return SECONDS_IN_DAY;
    }

    if (unit == "w" || unit == "week") {
      return 7 * SECONDS_IN_DAY;
    }

    throw std::invalid_argument(
        "\"" + unit +
        "\" is not a valid time unit. Valid time units: "
        "\"s\"/\"second\", \"m\"/\"minute\", \"h\"/\"hour\", \"d\"/\"day\", "
        "\"w\"/\"week\".");
  }

  uint32_t _in_seconds;

  // Private default constructor for cereal.
  Duration() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_in_seconds);
  }
};

}  // namespace thirdai::utils
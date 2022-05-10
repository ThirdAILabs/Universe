#pragma once

#include <cstdint>
#include <string>
#include <istream>
#include <sstream>
#include <iomanip>

namespace thirdai::dataset {

inline static uint32_t getNumberU32(const std::string& str) {
  const char* start = str.c_str();
  char* end;
  return std::strtoul(start, &end, 10);
}

inline static std::tm getTm(std::string& str, const std::string& timestamp_fmt) {
  std::tm t = {};
  std::istringstream ss(str, std::ios_base::in);
  ss >> std::get_time(&t, timestamp_fmt.c_str());
  return t;
}

inline static uint32_t getSecondsSinceEpochU32(const std::string& str, const std::string& timestamp_fmt) {
  std::tm t = {};
  std::istringstream ss(str, std::ios_base::in);
  ss >> std::get_time(&t, timestamp_fmt.c_str());
  return mktime(&t);
}

} // namespace thirdai::dataset
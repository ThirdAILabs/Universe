#pragma once

#include "Schema.h"
#include <schema/InProgressVector.h>
#include <cstddef>
#include <ctime>

namespace thirdai::schema {

struct DateBlock: public ABlock {
  DateBlock(const uint32_t col, const uint32_t offset, std::string timestamp_fmt, uint32_t n_years): _col(col), _offset(offset), _n_years(n_years), _timestamp_fmt(std::move(timestamp_fmt)) {}

  void extractFeatures(std::vector<std::string_view> line, InProgressSparseVector &vec) override {
    std::tm time = getTm(line[_col], _timestamp_fmt);
    uint32_t day_of_week = time.tm_wday; // 0 to 6; 7 possible values
    uint32_t day_of_month = time.tm_mday - 1; // -1 for 0-indexing; 0 to 30; 31 possible values
    uint32_t month_of_year = time.tm_mon; // 0 to 11; 12 possible values
    uint32_t week_of_month = day_of_month / 7; // 0 to 30/7; 0 to 4; 5 possible values
    uint32_t day_of_year = time.tm_yday; // 0 to 365; 366 possible values
    uint32_t year = time.tm_year % _n_years; // 0 to _n_years - 1; _n_years possible values.
    
    auto offset = _offset;
    vec.addSingleFeature(offset + year, 1);
    offset += _n_years;
    vec.addSingleFeature(_offset + day_of_week, 1);
    offset += _n_days_in_week;
    vec.addSingleFeature(offset + day_of_month, 1);
    offset += _n_days_in_month;
    vec.addSingleFeature(offset + week_of_month, 1);
    offset += _n_weeks_in_month;
    vec.addSingleFeature(offset + month_of_year, 1);
    offset += _n_months_in_year;
    vec.addSingleFeature(offset + day_of_year, 1);
  }
    
  static std::shared_ptr<ABlockConfig> Config(const uint32_t col, std::string timestamp_fmt, uint32_t n_years=10) {
    return std::make_shared<DateBlockConfig>(col, std::move(timestamp_fmt), n_years);
  }
  struct DateBlockConfig: public ABlockConfig {
    explicit DateBlockConfig(const uint32_t col, std::string timestamp_fmt, uint32_t n_years): _col(col), _n_years(n_years), _timestamp_fmt(std::move(timestamp_fmt)) {}

    std::unique_ptr<ABlock> build(uint32_t &offset) const override {
      auto built = std::make_unique<DateBlock>(_col, offset, _timestamp_fmt, _n_years);
      offset += (_n_years + _out_dim_excl_years);
      return built;
    }

    size_t maxColumn() const override { return _col; }

    size_t featureDim() const override { return _n_years + _out_dim_excl_years; }

  private:
    uint32_t _col;
    uint32_t _n_years;
    uint32_t _out_dim_excl_years = _n_days_in_week + _n_days_in_month + _n_weeks_in_month + _n_months_in_year + _n_days_in_year;
    std::string _timestamp_fmt;
  };

 private:

  uint32_t _col;
  uint32_t _offset;
  uint32_t _n_years;
  std::string _timestamp_fmt;
  static constexpr uint32_t _n_days_in_week = 7;
  static constexpr uint32_t _n_days_in_month = 31;
  static constexpr uint32_t _n_weeks_in_month = 5;
  static constexpr uint32_t _n_months_in_year = 12;
  static constexpr uint32_t _n_days_in_year = 366;
};


} // namespace thirdai::schema




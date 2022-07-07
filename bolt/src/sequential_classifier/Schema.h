#include <dataset/src/blocks/BlockInterface.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

struct Item {
  std::string col_name;
  uint32_t n_distinct;
  dataset::GraphPtr graph = nullptr;
  uint32_t max_neighbors = 0;
  uint32_t col_num = 0;
};

struct CategoricalAttribute {
  std::string col_name;
  uint32_t n_distinct;
  uint32_t col_num = 0;
};

struct Timestamp {
  std::string col_name;
  uint32_t col_num = 0;
};

struct TrackingConfig {
  uint32_t horizon;
  uint32_t lookback;
  uint32_t period = 1;
};

struct TrackableQuantity {
  std::string col_name;
  bool has_col_num = false;
  uint32_t col_num = 0;
};

struct TextAttribute {
  std::string col_name;
  uint32_t col_num = 0;
};

struct Schema {
 public:
  Schema(Item item, Timestamp timestamp, CategoricalAttribute target,
         TrackingConfig tracking_config,
         std::vector<TextAttribute> text_attrs = {},
         std::vector<CategoricalAttribute> cat_attrs = {},
         std::vector<TrackableQuantity> trackable_qtys = {})
      : item(std::move(item)),
        timestamp(std::move(timestamp)),
        target(std::move(target)),
        tracking_config(tracking_config),
        text_attributes(std::move(text_attrs)),
        categorical_attributes(std::move(cat_attrs)),
        trackable_quantities(std::move(trackable_qtys)) {}

  void fitToHeader(const std::string& header, char delimiter) {
    auto name_to_num = buildNameToNumMap(header, delimiter);

    setColNum(name_to_num, item.col_name, item.col_num);
    setColNum(name_to_num, timestamp.col_name, item.col_num);
    for (auto& text : text_attributes) {
      setColNum(name_to_num, text.col_name, text.col_num);
    }
    for (auto& cat : categorical_attributes) {
      setColNum(name_to_num, cat.col_name, cat.col_num);
    }
    for (auto& track : trackable_quantities) {
      if (track.col_name.empty()) {
        track.has_col_num = false;
        track.col_num = 0;
      } else {
        track.has_col_num = true;
        setColNum(name_to_num, track.col_name, track.col_num);
      }
    }
  }

  static std::unordered_map<std::string, uint32_t> buildNameToNumMap(
      const std::string& header, char delimiter) {
    std::unordered_map<std::string, uint32_t> name_to_num;
    uint32_t col = 0;
    size_t start = 0;
    size_t end = 0;
    while (end != std::string::npos) {
      end = header.find(delimiter, start);
      size_t len =
          end == std::string::npos ? header.size() - start : end - start;
      name_to_num[header.substr(start, len)] = col;
      col++;
      start = end + 1;
    }
    return name_to_num;
  }

  static void setColNum(std::unordered_map<std::string, uint32_t> name_to_num,
                        const std::string& col_name, uint32_t& col_num) {
    if (name_to_num.count(col_name) == 0) {
      std::stringstream error_ss;
      error_ss << "Could not find a column named '" << col_name
               << "' in the header.";
      throw std::invalid_argument(error_ss.str());
    }
    col_num = name_to_num.at(col_name);
  }

  Item item;
  Timestamp timestamp;
  CategoricalAttribute target;
  TrackingConfig tracking_config;
  std::vector<TextAttribute> text_attributes;
  std::vector<CategoricalAttribute> categorical_attributes;
  std::vector<TrackableQuantity> trackable_quantities;
};

}  // namespace thirdai::bolt
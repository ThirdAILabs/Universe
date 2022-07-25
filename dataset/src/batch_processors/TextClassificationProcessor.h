#pragma once

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/encodings/text/TextEncodingUtils.h>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace thirdai::dataset {

class TextClassificationProcessor final : public UnaryBoltBatchProcessor {
 public:
  explicit TextClassificationProcessor(uint32_t output_range)
      : _output_range(output_range) {}

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {
    auto [lhs, rhs] = split(header);

    if (lhs == "text" && rhs == "category") {
      _label_on_right = true;
    } else if (lhs == "category" && rhs == "text") {
      _label_on_right = false;
    } else {
      throw std::invalid_argument("Invalid column names '" + std::string(lhs) +
                                  "' and '" + std::string(rhs) +
                                  "'. Expected 'category' and 'text'.");
    }
  }

  std::string getClassName(uint32_t class_id) const {
    return _class_id_to_class.at(class_id);
  }

  std::vector<std::string> getClassIdToNames() { return _class_id_to_class; }

 protected:
  std::pair<bolt::BoltVector, bolt::BoltVector> processRow(
      const std::string& row) final {
    // Split the row
    auto [lhs, rhs] = split(row);

    if (_label_on_right) {
      bolt::BoltVector label_vec = getLabel(rhs);
      bolt::BoltVector data_vec =
          TextEncodingUtils::computePairgrams(lhs, _output_range);
      return std::make_pair(std::move(data_vec), std::move(label_vec));
    }
    bolt::BoltVector label_vec = getLabel(lhs);
    bolt::BoltVector data_vec =
        TextEncodingUtils::computePairgrams(rhs, _output_range);
    return std::make_pair(std::move(data_vec), std::move(label_vec));
  }

 private:
  bolt::BoltVector getLabel(std::string_view category_str_view) {
    std::string category_str(category_str_view);
    uint32_t label;
    if (_class_to_class_id.count(category_str)) {
      label = _class_to_class_id[category_str];
    } else {
      label = _class_id_to_class.size();
      _class_to_class_id[category_str] = label;
      _class_id_to_class.push_back(std::move(category_str));
    }

    bolt::BoltVector label_vec(1, false, false);
    label_vec.active_neurons[0] = label;
    label_vec.activations[0] = 1.0;

    return label_vec;
  }

  static bool isQuote(char c) { return c == '"' || c == '\''; }

  static std::string_view trim(std::string_view& str) {
    uint32_t start_offset = 0;
    while (start_offset < (str.size() - 1) &&
           (std::isspace(str[start_offset]) || isQuote(str[start_offset]))) {
      start_offset++;
    }

    uint32_t end_offset = str.size();
    while (end_offset > 0 && (std::isspace(str[end_offset - 1]) ||
                              isQuote(str[end_offset - 1]))) {
      end_offset--;
    }

    return std::string_view(str.data() + start_offset,
                            end_offset - start_offset);
  }

  std::pair<std::string_view, std::string_view> split(
      const std::string& line) const {
    std::string::size_type split_index;
    if (_label_on_right) {
      split_index = line.find_last_of(',');
    } else {
      split_index = line.find(',');
    }
    if (split_index == std::string::npos) {
      throw std::invalid_argument("No comment in line '" + line +
                                  "' of csv file");
    }
    std::string_view lhs = std::string_view(line.data(), split_index);
    std::string_view rhs = std::string_view(line.data() + split_index + 1,
                                            line.size() - split_index - 1);

    if (lhs.empty() || rhs.empty()) {
      throw std::invalid_argument(
          "Line '" + line +
          "' is improperly formatted. Expected <label>,<text> or "
          "<text>,<label>. Either the label or the text is the empty string.");
    }
    lhs = trim(lhs);
    rhs = trim(rhs);

    if (lhs.empty() || rhs.empty()) {
      throw std::invalid_argument(
          "Line '" + line +
          "' is improperly formatted. Expected <label>,<text> or "
          "<text>,<label>. Either the label or the text is the empty string "
          "after removing quotes.");
    }

    return {lhs, rhs};
  }

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<UnaryBoltBatchProcessor>(this),
            _class_to_class_id, _class_id_to_class, _output_range,
            _label_on_right);
  }

  // Private constructor for cereal.
  TextClassificationProcessor() {}

  std::unordered_map<std::string, uint32_t> _class_to_class_id;
  std::vector<std::string> _class_id_to_class;
  uint32_t _output_range;

  bool _label_on_right;
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TextClassificationProcessor)
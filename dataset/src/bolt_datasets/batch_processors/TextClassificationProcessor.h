#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <dataset/src/bolt_datasets/BatchProcessor.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace thirdai::dataset {

class TextClassificationProcessor final : public UnaryBatchProcessor {
 public:
  std::string getClassName(uint32_t class_id) const {
    return _class_id_to_class.at(class_id);
  }

  static bolt::BoltVector computePairGramHashes(std::string_view sentence) {
    std::vector<uint32_t> hashes_seen;
    std::unordered_map<uint32_t, uint32_t> pairgram_hashes;
    bool prev_is_space = true;
    uint32_t start_of_word_offset;
    for (uint32_t i = 0; i < sentence.size(); i++) {
      if (prev_is_space && !std::isspace(sentence[i])) {
        start_of_word_offset = i;
        prev_is_space = false;
      }
      if (!prev_is_space && std::isspace(sentence[i])) {
        uint32_t len = i - start_of_word_offset;
        uint32_t hash =
            hashing::MurmurHash(sentence.data() + start_of_word_offset, len,
                                /* seed = */ 3829);
        for (const uint32_t prev_hash : hashes_seen) {
          pairgram_hashes[hashing::HashUtils::combineHashes(prev_hash, hash)]++;
        }
        hashes_seen.push_back(hash);

        prev_is_space = true;
      }
    }

    bolt::BoltVector data_vec(pairgram_hashes.size(), false, false);
    uint32_t index = 0;
    for (auto& entry : pairgram_hashes) {
      data_vec.active_neurons[index] = entry.first;
      data_vec.activations[index] = entry.second;
      index++;
    }

    return data_vec;
  }

 protected:
  std::pair<bolt::BoltVector, bolt::BoltVector> processRow(
      const std::string& row) final {
    // Find the label
    uint32_t label;
    uint32_t end_of_label = row.find(',');
    // TODO(nicholas): Trim this?
    std::string label_str = row.substr(0, end_of_label);
    if (_class_to_class_id.count(label_str)) {
      label = _class_to_class_id[label_str];
    } else {
      label = _class_id_to_class.size();
      _class_to_class_id[label_str] = label;
      _class_id_to_class.push_back(std::move(label_str));
    }

    bolt::BoltVector label_vec(1, false, false);
    label_vec.active_neurons[0] = label;
    label_vec.activations[0] = 1.0;

    std::string_view str(row.data() + end_of_label + 1,
                         row.size() - end_of_label + 1);

    // Compute pair gram hashes
    bolt::BoltVector data_vec = computePairGramHashes(str);

    return std::make_pair(std::move(label_vec), std::move(data_vec));
  }

 private:
  std::unordered_map<std::string, uint32_t> _class_to_class_id;
  std::vector<std::string> _class_id_to_class;
};

}  // namespace thirdai::dataset
#include "GlobalFreq.h"

namespace thirdai::utils::dataset {

GlobalFreq::GlobalFreq(std::unique_ptr<StringLoader> string_loader,
                       const vectorizer_config_t& vectorizer_config,
                       std::vector<std::string>& filenames)
    : _vectorizer_config(vectorizer_config) {
  std::unordered_set<uint32_t> temp_set;
  std::unordered_map<uint32_t, float> empty_map;
  size_t total_doc_count = 0;
  std::vector<std::string> loaded_strings;
  std::vector<std::vector<uint32_t>> loaded_labels;
  std::unordered_map<uint32_t, float> index_to_value_map;
  CompositeVectorizer composite_vectorizer(vectorizer_config);
  for (auto const& file : filenames) {
    std::ifstream fstream(file);
    // Load one vector at a time to get document frequency
    // batch_size is used to create a one-string-at-a-time streaming setting
    uint32_t batch_size = 1;
    // Keep loading until the whole file is loaded
    while (string_loader->loadStringsAndLabels(fstream, batch_size,
                                               loaded_strings,
                                               loaded_labels) == batch_size) {
      loaded_strings.clear();
      loaded_labels.clear();
      index_to_value_map.clear();
      total_doc_count++;

      for (auto const& string : loaded_strings) {
        composite_vectorizer.fillIndexToValueMap(string, index_to_value_map,
                                                 empty_map);
      }

      for (auto kv : index_to_value_map) {
        uint32_t hash = kv.first;
        if (temp_set.find(hash) == temp_set.end()) {
          temp_set.insert(hash);
          // keys in _idf_map have default values of 1
          _idf_map[hash]++;
        } else {
          // std::cout << "Clashed  word:" << temp << "  Sentence: " << buffer
          // << std::endl;
        }
      }
      temp_set.clear();
    }
  }
  for (auto& kv : _idf_map) {
    float df = kv.second;
    float idf = std::log(total_doc_count / df);
    _idf_map[kv.first] = idf;
  }
}

}  // namespace thirdai::utils::dataset
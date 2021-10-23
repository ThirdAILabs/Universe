#include "GlobalFreq.h"

namespace thirdai::utils::dataset {

GlobalFreq::GlobalFreq(std::unique_ptr<StringLoader> string_loader,
             vectorizer_config_t vectorizer_config,
             std::vector<std::string>&& filenames)
             : _vectorizer_config(vectorizer_config) {
  std::unordered_set<int> temp_set;
  size_t file_count = 0;
  size_t total_doc_count = 0;
  std::vector<std::string> loaded_strings;
  std::vector<std::vector<uint32_t>> loaded_labels;
  for (auto file:filenames) {
      std::ifstream fstream(file);
      // When I use StringLoader, how do I know when it has loaded to the end of file
      uint32_t batch_size = 100; // Temporary for now
      string_loader->loadStringsAndLabels(fstream, batch_size, loaded_strings, loaded_labels);
      
      loaded_strings.clear();
      loaded_labels.clear();
  }

}


}  // namespace thirdai::utils::dataset
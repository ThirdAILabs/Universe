#include "TfidfDataset.h"

namespace thirdai::utils {

void TfidfDataset::readDataset(const std::vector<std::string>& filenames,
                               TFIDF_FILE_TYPE fileType, TOKEN_TYPE tokenType) {
  (void)tokenType;
  // if (tokenType != TOKEN_TYPE::UNI_GRAM) { _idfDefault = 1; }
  int counter = 0;
  std::unordered_map<std::string, int>
      tfMap;  // TODO: Function to calculate TF(w,d)
  // stream and read
  // TODO: Use a switch statement
  if (fileType == TFIDF_FILE_TYPE::FILE) {
    // TODO: We should parallelize this so an array would be better.
    for (auto const& filename : filenames) {
      std::ifstream file(filename);
      if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
        throw std::runtime_error("Unable to open file '" + filename + "'");
      }
      // Start reading each file
      std::string line;
      // We stream the tokens in the file
      // Word unigram. TODO: Should write in a way that can tokenize a
      // sentence/paragraph according to token type so that we dont have to
      // write separate cases for unigram and trigram

      while (
          std::getline(file, line)) {  // Read in until a "\n". TODO: Use
                                       // Geordie's method to read in a sentence
        std::stringstream stream(line);
        std::string str;
        // The following loop can be parallel
        while (stream >> str) {
          _indices.push_back(_tokenIdMap[str]);  // This tokenIdMap is just a
                                                 // placeholder for indices
          _values.push_back(
              tfMap[str] *
              _globalFreq->getIdf(
                  str));  // Default value is taken care of in getIdf
          _markers.push_back(counter);
          counter++;
        }
        line.clear();
      }
      tfMap.clear();
    }
  }
}

}  // namespace thirdai::utils
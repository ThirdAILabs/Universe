#include "TfidfDataset.h"

namespace thirdai::utils{

void TfidfDataset::readDataset(std::vector<std::string> filenames, TFIDF_FILE_TYPE fileType, TOKEN_TYPE tokenType) {
    //if (tokenType != TOKEN_TYPE::UNI_GRAM) { _idfDefault = 1; }
    int counter = 0;
    std::unordered_map<std::string, int> map;
    // stream and read
    if (fileType == TFIDF_FILE_TYPE::FILE) {
         for (auto filename : filenames) {
            std::ifstream file(filename);
            if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
                throw std::runtime_error("Unable to open file '" + filename + "'");
            }
            // Start reading each file
            std::string line;
            // We stream the tokens in the file
            // word unigram
            while (std::getline(file, line)) { // Read in until a "\n"
                std::stringstream stream(line);
                std::string str;
                while (stream >> str) {
                    
                    _indices.push_back(_tokenIdMap[str]);   // This tokenIdMap is just a placeholder for indices
                    _values.push_back( map[str] * _globalFreq->getIdf(token)); // Default value is taken care of in getIdf
                    _markers.push_back(counter);
                    counter++;
                }
                line.clear();
            }

         }
    }
}


} // namespace thirdai::utils
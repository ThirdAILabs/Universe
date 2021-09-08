#include "TfidfDataset.h"

namespace thirdai::utils{

void TfidfDataset::readDataset(std::vector<std::string> filenames, TFIDF_FILE_TYPE type) {
    
    // stream and read
    if (type == TFIDF_FILE_TYPE::FILE) {
         for (auto filename : filenames) {
            std::ifstream file(filename);
            if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
                throw std::runtime_error("Unable to open file '" + filename + "'");
            }
            // Start reading each file
            std::string line;
            // We stream the toekns in the file, and don't load the whole file into memory
            

         }
    }
}


} // namespace thirdai::utils
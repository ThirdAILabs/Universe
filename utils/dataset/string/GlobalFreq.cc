#include "GlobalFreq.h"

namespace thirdai::utils {

GlobalFreq::GlobalFreq(std::vector<std::string>& files, StringLoader* string_loader, u_int32_t murmur_seed) {

    _string_loader = string_loader;
    _murmur_seed = murmur_seed;
    // start building the IDF map
    size_t file_count = 0;
    for (auto file : files) {
        std::string buffer;
        string_loader->updateFile(file);
        while (string_loader->loadNextString(buffer)) {
            file_count++;
            // Murmurhash them. We need unigram & bigram & trigram
            
        }
        
    }
}

GlobalFreq::~GlobalFreq() {}

}  // namespace thirdai::utils
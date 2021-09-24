#include "GlobalFreq.h"

namespace thirdai::utils {

GlobalFreq::GlobalFreq(std::vector<std::string>& files, StringLoader* string_loader, u_int32_t murmur_seed) {

    _string_loader = string_loader;
    _murmur_seed = murmur_seed;
    // start building the IDF map
    std::unordered_set<int> temp_set;
    size_t file_count = 0;
    for (auto file : files) {
        std::string buffer;
        string_loader->updateFile(file);
        while (string_loader->loadNextString(buffer)) {
            file_count++;
            // Murmurhash them. We need unigram & bigram & trigram
            // We will do unigram for V1
            std::stringstream stream(buffer);
            std::string temp;
            while (stream >> temp) {
                u_int32_t len = temp.length();
                const char *converted = temp.c_str();
                int hash = MurmurHash(converted, len, _murmur_seed);
                if (temp_set.find(hash) != temp_set.end()) {
                    _idfMap[hash] += 1;
                }
                else {
                    temp_set.insert(hash);
                }
            }
            temp_set.clear();
        }
        
    }
}

GlobalFreq::~GlobalFreq() {}

}  // namespace thirdai::utils
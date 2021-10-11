#include "GlobalFreq.h"

namespace thirdai::utils {

GlobalFreq::GlobalFreq(std::vector<std::string>& files, StringLoader* string_loader, u_int32_t murmur_seed) {

    _string_loader = string_loader;
    _murmur_seed = murmur_seed;
    // start building the IDF map
    std::unordered_set<int> temp_set;
    size_t file_count = 0;
    size_t total_doc_count = 0;
    for (auto file : files) {
        std::string buffer;
        // TODO: (henry) Make sure we really need this line because StringDataset already calls it
        string_loader->addFileToQueue(file);
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
                // std::cout << temp << " " << hash << "  ";
                if (temp_set.find(hash) == temp_set.end()) {
                    temp_set.insert(hash);
                    if (_idfMap.count(hash) > 0) {
                        _idfMap[hash]++;
                    }
                    else {
                        _idfMap[hash] = 1;
                    }
                }
                else {
                    // std::cout << "Clashed  " << temp << std::endl;
                }
            }
            temp_set.clear();
            total_doc_count ++;
        }
    }
    for (auto &kv : _idfMap) {
        float df = kv.second;
        float idf = log(total_doc_count/df);
        _idfMap[kv.first] = idf;
    }
}

int GlobalFreq::idf_size() {
    return _idfMap.size();
}

int GlobalFreq::getIdf(std::string& token) {
    u_int32_t len = token.length();
    const char *converted = token.c_str();
    int hash = MurmurHash(converted, len, _murmur_seed);
    if (_idfMap.count(hash) > 0) {
        return _idfMap[hash];
    }
    else {
        return 0;
    }
}

// GlobalFreq::~GlobalFreq() {}

}  // namespace thirdai::utils
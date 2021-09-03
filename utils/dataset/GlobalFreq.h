#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace thirdai::utils {
    
class GlobalFreq
{
    /*
        This class is responsible for calculating (inverse) word/document frequencies on a given data corpus. 
    */
    private:
        std::unordered_map<std::string, int> _idfMap;
    
    public:
        GlobalFreq(std::string[]);

        inline int getIdf(std::string word);

        // Can have a parallel version of getIdf like:
        void getIdfPar(std::vector<std::string> wordvec, int* freqs);

        ~GlobalFreq();
    };
    
    GlobalFreq::GlobalFreq(std::string files[]) {
    }
    
    GlobalFreq::~GlobalFreq() {
    }
    
}
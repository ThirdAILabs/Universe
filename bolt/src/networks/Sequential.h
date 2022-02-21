#pragma once

#include <layers/Layer.h>
#include <Network.h>
#include <vector>

namespace thirdai::bolt {

class Sequential: public Network {
    public:
        Sequential(std::vector<Layer>&& layers, uint32_t input_dim);
};

}  // namespace thirdai::bolt
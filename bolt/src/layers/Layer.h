#pragma once

#include <BoltVector.h>

namespace thirdai::bolt {

class Layer {
    public:
        void forward(BoltVector& input, BoltVector& output);

        void backpropagate(BoltVector& input, BoltVector& output);

        void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);
};

}  // namespace thirdai::bolt
#pragma once

#include "../../utils/dataset/batch_types/ClickThroughBatch.h"
#include "../layers/EmbeddingLayer.h"
#include "../layers/FullyConnectedLayer.h"
#include <vector>

namespace thirdai::bolt {

class TwoTower {

public:
    

private:
    uint32_t _num_layers;
    FullyConnectedLayer** _model1_layers;
    FullyConnectedLayer** _model2_layers;

    std::vector<FullyConnectedLayerConfig> _model1_layer_configs;
    std::vector<FullyConnectedLayerConfig> _model2_layer_configs;

};


}  // namespace thirdai::bolt
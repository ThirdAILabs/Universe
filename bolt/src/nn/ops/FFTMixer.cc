#include "FFTMixer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <fftw3.h>


namespace thirdai::bolt::nn::ops {

void FFTMixer::forward(const autograd::ComputationList& inputs,
                   tensor::TensorPtr& output, uint32_t index_in_batch,
                   bool training) {
    (void)inputs;
    (void)output;
    (void)index_in_batch;
    (void)training;
}

void FFTMixer::backpropagate(autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch){
    (void)inputs;
    (void)output;
    (void)index_in_batch;
}

void FFTMixer::updateParameters(float learning_rate, uint32_t train_steps){
    (void)learning_rate;
    (void)train_steps;
}

} // namespace thirdai::bolt::nn::ops

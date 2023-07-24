#pragma once


#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class FFT final: public Op,
                public std::enable_shared_from_this<FFT> {
private:
    FFT();
public:
    static auto make(){
        return std::shared_ptr<FFT>(new FFT());
    }
    void forward(const autograd::ComputationList& inputs,
               tensor::TensorPtr& output, uint32_t index_in_batch,
               bool training) final;

    void backpropagate(autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch) final;

    void updateParameters(float learning_rate, uint32_t train_steps) final;
    
    uint32_t dim() const final { return _dim; }

     void disableSparseParameterUpdates() final {
        _disable_sparse_parameter_updates = true;
    }

    void enableSparseParameterUpdates() final {
        _disable_sparse_parameter_updates = false;
    }


    std::vector<std::vector<float>*> gradients() final {
        return {};
    }

    std::vector<std::vector<float>*> parameters() final {
        return {};
    }

    void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

    std::optional<uint32_t> nonzeros(const autograd::ComputationList& inputs,
                                   bool use_sparsity) const final {
        (void)inputs;
        (void)use_sparsity;
        return dim();
    }
    
private:
    bool _disable_sparse_parameter_updates;
    uint32_t _dim, _prev_dim;

};
using FFTPtr = std::shared_ptr<FFT>;
}  // namespace thirdai::bolt::nn::ops
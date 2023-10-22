#pragma once

#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/autograd/ComputationGraph.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt {
class FFTMixer final : public Op, public std::enable_shared_from_this<FFTMixer> {
  public:
  static std::shared_ptr<FFTMixer> make(uint32_t row, uint32_t columns);
  
  void forward(const ComputationList& inputs,
               TensorPtr& output, uint32_t index_in_batch,
               bool training) final;

  void backpropagate(ComputationList& inputs,
                     TensorPtr& output, uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final { return _rows * _columns; };

  void disableSparseParameterUpdates() final {}

  void enableSparseParameterUpdates() final {}

  void initOptimizer() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; }

  std::vector<std::vector<float>*> parameters() final { return {}; }

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final {
    return inputs.at(0)->nonzeros(use_sparsity);
  }

  uint32_t inputDim() const { return _rows * _columns; }
  /**Applies the op to an input tensor and yields
          a new output tensor.Used to* add the op to a computation graph. */
    ComputationPtr apply(ComputationPtr input);

 private:
  FFTMixer() {}

  FFTMixer(uint32_t rows, uint32_t columns);

  bool _disable_sparse_parameter_updates;
  uint32_t _rows, _columns;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};
using FFTMixerPtr = std::shared_ptr<FFTMixer>;

}  // namespace thirdai::bolt
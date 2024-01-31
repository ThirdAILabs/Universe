#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <valarray>

namespace thirdai::bolt {

class Computation;

using ComputationPtr = std::shared_ptr<Computation>;
using ComputationList = std::vector<ComputationPtr>;

class Model;

/**
 * Represents a operation in a model which takes in one or more inputs and
 * produces an output. The tensors which store the values for the inputs and
 * output should not be stored or referenced by the op and will be passed into
 * the op for forward and backpropagate. This is so the op is not fixed to a
 * particular location in a model and can be reused. For instance if you want to
 * use a FullyConnected op in a simple RNN, then the op itself is the same, but
 * the inputs and output it is called with changes. The op should store and
 * maintain any parameters it uses, for example a weight matrix or embedding
 * table, and it can store information about expected dimensions that are
 * invariant across different inputs it may be called with.
 */
class Op {
 public:
  explicit Op(std::string name) : _name(std::move(name)) {}

  /**
   * Computes the forward computation of the op. This should use the inputs in
   * the given set of input tensors and store the result in the given output
   * tensor. The parameter index_in_batch indicates which sample of the batch
   * the op should process. This allows the model to parallelize the entire
   * forward and/or backward pass through the graph across the batch. Thus, this
   * function should be thread safe to being called with different values for
   * index_in_batch at the same time.
   */
  virtual void forward(const ComputationList& inputs, TensorPtr& output,
                       uint32_t index_in_batch, bool training) = 0;

  /**
   * Computes the gradients of any parameters in the op and with respect to the
   * output tensor, and backpropagates gradients from the output tensor to the
   * inputs. The op should increment the gradients of its inputs so that it does
   * not overwrite existing gradients other ops may compute for the same
   * tensor. The model ensures that backpropagate will will not be called on a
   * given op and its output until all the ops that backpropagate gradients to
   * that output have done so. The parameter index_in_batch indicates which
   * sample of the batch the op should process. This allows the model to
   * parallelize the entire forward and/or backward pass through the graph
   * across the batch. Thus, this function should be thread safe to being called
   * with different values for index_in_batch at the same time (though benign
   * race conditions to e.g. weight array are sometimes okay for performance).
   */
  virtual void backpropagate(ComputationList& inputs, TensorPtr& output,
                             uint32_t index_in_batch) = 0;

  virtual void updateTrainableParameters(float learning_rate,
                                         uint32_t train_steps) {
    if (_trainable) {
      updateParameters(learning_rate, train_steps);
    }
  }

  /**
   * Performs a parameter update on any parameters in the op. The parameter
   * train steps represents how many train steps have been completed so far in
   * the model. This is for logging and also optimizers like Adam which requires
   * this for bias correction.
   */
  virtual void updateParameters(float learning_rate, uint32_t train_steps) = 0;

  /**
   * Returns the output dimension of the op. Does not include batch size. For
   * instance a fully connected layer op will return its number of neurons.
   */
  virtual uint32_t dim() const = 0;

  /**
   * Returns the number of nonzeros in the ops output for a given list of
   * inputs that will be passed to the op's forward computation. Returns
   * std::nullopt if the number of nonzeros is not fixed. The use_sparsity
   * argument indicates whether sparsity will be used in the forward
   * computation. There are several reasons why the number of nonzeros may be
   * dependent on these arguments. If the op is a FullyConnected op using a
   * sparse layer the number of nonzeros in the output will be dependent on
   * whether or not sparsity is being used. If the op is a Concatenate op with
   * sparse inputs, then if sparsity is being used the number of nonzeros in the
   * output will depend on the number of nonzeros in the inputs.
   */
  virtual std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                           bool use_sparsity) const = 0;

  /**
   * Initializes the optimizer for the op.
   */
  virtual void initOptimizer() = 0;

  /**
   * Disables sparse parameter updates for updateParameters in the op. This is
   * used for distributed and also can be beneficial in cases where most of the
   * parameters are being updated and dense updates are faster.
   */
  virtual void disableSparseParameterUpdates() = 0;

  /**
   * Enables sparse parameter updates for updateParameters in the op. This is
   * used for distributed to enable sparse updates once distributed training is
   * complete.
   */
  virtual void enableSparseParameterUpdates() = 0;

  /**
   * Returns references to all of the gradients of the op. Used for distributed
   * training.
   */
  virtual std::vector<std::vector<float>*> gradients() = 0;
  /**
   * Returns references to all of the weights of the op. Used for distributed
   * training.
   */
  virtual std::vector<std::vector<float>*> parameters() = 0;

  /**
   * Appends a line to the summary to describe the op when applied to the given
   * inputs and yielding the given output. Ideally this should be in the form:
   * OpType(op name): input(s) -> output(s) [op parameters]
   */
  virtual void summary(std::ostream& summary, const ComputationList& inputs,
                       const Computation* output) const = 0;

  /**
   * Controls if the op should save the optimizer along with the parameters.
   */
  virtual void setSerializeOptimizer(bool setSerializeOptimizer) {
    (void)setSerializeOptimizer;
  }

  virtual void switchToSgd() {}

  virtual void registerModel(const std::weak_ptr<Model>& model) { (void)model; }

  virtual std::vector<std::pair<std::string, double>> parameterAndGradNorms()
      const {
    return {};
  }

  /**
   * Returns the name of the op. All of the ops in a model must have a
   * unique name.
   */
  const std::string& name() const { return _name; }

  void setName(std::string name) { _name = std::move(name); }

  void setTrainable(bool flag) { _trainable = flag; }

  bool isTrainable() const { return _trainable; }

  virtual ~Op() = default;

 protected:
  Op() : Op("unnamed-op") {}

  static std::tuple<double, double, double> norms(const float* data,
                                                  size_t len) {
    double l1_norm = 0;
    double l2_norm = 0;
    double l_inf_norm = 0;

    for (size_t i = 0; i < len; i++) {
      l1_norm += std::abs(data[i]);
      l2_norm += data[i] * data[i];
      l_inf_norm = std::max<double>(l_inf_norm, std::abs(data[i]));
    }

    return {l1_norm, std::sqrt(l2_norm), l_inf_norm};
  }

  static void computeNorms(
      const std::vector<float>& data, const std::string& prefix,
      std::vector<std::pair<std::string, double>>& all_norms) {
    auto [l1_norm, l2_norm, l_inf_norm] = norms(data.data(), data.size());
    all_norms.emplace_back(prefix + "_l1_norm", l1_norm);
    all_norms.emplace_back(prefix + "_l2_norm", l2_norm);
    all_norms.emplace_back(prefix + "_l_inf_norm", l_inf_norm);
  }

 private:
  std::string _name;
  bool _trainable = true;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_name);
  }
};

using OpPtr = std::shared_ptr<Op>;

}  // namespace thirdai::bolt
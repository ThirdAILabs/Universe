#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/AllocationManager.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <licensing/src/CheckLicense.h>
#include <licensing/src/entitlements/TrainPermissionsToken.h>
#include <utils/UUID.h>
#include <vector>

namespace thirdai::bolt::nn::model {

/**
 * This represents the core Model that the Bolt engine is based around. This
 * combines the logic in the computations, loss functions, and autograd. These
 * Models are designed to only support processing a single batch at a time. This
 * simplifies the code and allows for things such as metrics, callbacks,
 * validation etc. to be delegated to a seperate trainer class. Additionally
 * this allows for users to interact with the model at a lower level if needed
 * to write custom training functions and interact with the state of the model
 * after processing batches.
 */
class Model {
 public:
  Model(autograd::ComputationList inputs, autograd::ComputationList outputs,
        std::vector<loss::LossPtr> losses);

  static std::shared_ptr<Model> make(autograd::ComputationList inputs,
                                     autograd::ComputationList outputs,
                                     std::vector<loss::LossPtr> losses);

  /**
   * Computes the forward pass through the model for the given batch.
   * Activations are not cleared until the next call to forward or trainOnBatch.
   * This is intended for use with inference/evaluation and so labels will not
   * be selected by sparse fully connected layers which yield outputs since
   * labels are not provided. For this type of sampling trainOnBatch must be
   * used.
   */
  tensor::TensorList forward(const tensor::TensorList& inputs,
                             bool use_sparsity);

  /**
   * Performs the foward and backward pass through the model for the given
   * training batch. There is no intermediate thread synchronization between ops
   * or forward/backpropagate. Does not perform parameter updates. Labels will
   * be selected by sparse fully connected layers which yield outputs.
   */
  void trainOnBatch(const tensor::TensorList& inputs,
                    const tensor::TensorList& labels);

  /**
   * Performs the forward pass through the model on a given batch. Differs from
   * forward methods above because the labels can be provided so that they are
   * set and can thus be used in metrics during validation. The labels are not
   * used at all during the forward pass (i.e. not used in sparse layer
   * sampling) or anywhere else in the model.
   */
  tensor::TensorList forward(const tensor::TensorList& inputs,
                             const tensor::TensorList& labels,
                             bool use_sparsity);

  /**
   * Updates the parameters in the model. Loops through all ops in the model
   * and calls updateParameters on each so that each op performs one
   * optimizer step on all of its trainable parameters.
   */
  void updateParameters(float learning_rate);

  /**
   * Returns the list of ops in the model in the order they will be executed
   * during the forward pass. If an op is used multiple times in a model then it
   * will occur multiple times in this list, with the locations depending on
   * where the computations it is used in are located in the computation order.
   */
  std::vector<ops::OpPtr> opExecutionOrder() const;

  /**
   * Returns the list of computations in the model in the order they will be
   * computed during the forward pass. Differs from opComputationOrder because
   * ops may be reused by all computations are unique.
   */
  autograd::ComputationList computationOrder() const;

  /**
   * Returns the outputs of the model.
   */
  const autograd::ComputationList& outputs() const;

  /**
   * Returns the inputs storing the labels of the model.
   */
  const autograd::ComputationList& labels() const;

  /**
   * Returns the loss functions of the model.
   */
  const std::vector<loss::LossPtr>& losses() const;

  /**
   * Returns a list of all ops.
   */
  const std::vector<ops::OpPtr>& ops() const;

  /**
   * Retrieves on op by name. Throws if not found.
   */
  ops::OpPtr getOp(const std::string& name) const;

  /**
   * Retrieves a computation in the graph by name. Throws if not found.
   */
  autograd::ComputationPtr getComputation(const std::string& name) const;

  /**
   * Prints/returns a summary of the model. Throws if no op is found.
   */
  std::string summary(bool print = true) const;

  /**
   * Returns how many train steps the model has taken. Used for logging in
   * trainer.
   */
  uint32_t trainSteps() const;

  /**
   * Returns the dimensions of the inputs the model is expecting, in the order
   * they are expected.
   */
  std::vector<std::vector<uint32_t>> inputDims() const;

  /**
   * Returns the expected dimensions of the labels the model is expecting, in
   * the order they are expected.
   */
  std::vector<std::vector<uint32_t>> labelDims() const;

  /**
   * Returns a list of references to gradients of all parameters in the model.
   */
  std::vector<std::vector<float>*> gradients() const;

  /**
   * Freezes all hash tables in the model. The parameter
   * insert_labels_if_not_found controls if label neurons should be inserted
   * into the hash tables at the buckets that were probed when they are not
   * found during training.
   */
  void freezeHashTables(bool insert_labels_if_not_found);

  /**
   * Saves the model without optimizer state. Save metadata indicates if a
   * metadata file should also be created which gives the thirdai version, model
   * uuid, the date saved, number of train steps before the save, and the model
   * summary (summary only present if THIRDAI_EXPOSE_ALL is true).
   */
  void save(const std::string& filename, bool save_metadata = true);

  /**
   * Saves the model with optimizer state. Save metadata indicates if a
   * metadata file should also be created which gives the thirdai version, model
   * uuid, the date saved, number of train steps before the save, and the model
   * summary (summary only present if THIRDAI_EXPOSE_ALL is true).
   */
  void checkpoint(const std::string& filename, bool save_metadata = true);

  /**
   * Helper function to save the model to a stream.
   */
  void save_stream(std::ostream& output_stream) const;

  /**
   * Controls if the model will save the optimizer along with the parameters.
   */
  void setSerializeOptimizer(bool should_save_optimizer);

  /**
   * Loads the model and automatically initializes the optimizer state.
   */
  static std::shared_ptr<Model> load(const std::string& filename);

  static std::shared_ptr<Model> load_stream(std::istream& input_stream);

 private:
  /**
   * Computes the forward pass through the model for the given sample in the
   * batch. Assumes that setInputs(...) has already been called.
   */
  void forwardVector(uint32_t index_in_batch, bool training);

  /**
   * Computes the backward pass through the model for the given sample in the
   * batch. Assumes that setInputs(...) and setLabels(...) have already been
   * called.
   */
  void backpropagateVector(uint32_t index_in_batch, uint32_t batch_size);

  /**
   * Sets the given batch as the inputs to the model.
   */
  uint32_t setInput(const tensor::TensorList& input_batches);

  /**
   * Sets the given labels as the current labels for the model.
   */
  uint32_t setLabels(const tensor::TensorList& label_batches);

  /**
   * Returns a list of pairs of matching outputs and labels. A label and output
   * match if they are both used in a loss function with no other labels or
   * outputs.
   */
  std::vector<std::pair<autograd::ComputationPtr, autograd::ComputationPtr>>
  outputLabelPairs() const;

  /**
   * When a loss is applied to a single output computation coming from a fully
   * connected op this method connects that op with the corresponding labels in
   * the loss function so that the labels can be selected as active neurons when
   * the layer is sparse.
   */
  void matchOutputFullyConnectedLayersWithLabels() const;

  /**
   * Creates a metadata file which gives the thirdai version, model uuid, the
   * date saved, number of train steps before the save, and the model summary
   * (summary only present if THIRDAI_EXPOSE_ALL is true).
   */
  void saveMetadata(const std::string& save_path) const;

  void verifyAllowedOutputDim() const;

  autograd::ComputationList _inputs;
  autograd::ComputationList _outputs;
  autograd::ComputationList _labels;
  std::vector<loss::LossPtr> _losses;

  std::vector<ops::OpPtr> _ops;
  autograd::ComputationList _computation_order;

  AllocationManager _allocation_manager;

  uint32_t _train_steps;

  std::string _model_uuid;
  uint64_t _total_training_samples = 0;

  Model() : _allocation_manager() { licensing::checkLicense(); }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive, uint32_t version);
};

using ModelPtr = std::shared_ptr<Model>;

}  // namespace thirdai::bolt::nn::model
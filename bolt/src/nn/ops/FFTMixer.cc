#include "FFTMixer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/src/BoltVectorUtils.h>
#include <fftw3.h>

namespace thirdai::bolt{

std::string nextFFTOpName() {
  static uint32_t constructed = 0;
  return "fft_mixer_" + std::to_string(++constructed);
}

FFTMixer::FFTMixer(uint32_t rows, uint32_t columns)
    : Op(nextFFTOpName()), _rows(rows), _columns(columns) {}

std::shared_ptr<FFTMixer> FFTMixer::make(uint32_t rows, uint32_t columns) {
  return std::shared_ptr<FFTMixer>(new FFTMixer(rows, columns));
}


//Input to this op must be dense. add a check for it.
void FFTMixer::forward(const ComputationList& inputs,
                       TensorPtr& output, uint32_t index_in_batch,
                       bool training) {
  (void)training;
  assert(inputs.size() == 1 || inputs.size() == 2);
  // If the op is an output pass in labels during training to ensure labels are
  // in active neuron set.
  const BoltVector* labels = nullptr;
  if (training && inputs.size() == 2) {
    labels = &inputs[1]->tensor()->getVector(index_in_batch);
  }
  if (labels != nullptr) {
    throw std::logic_error("FFTMixers should not have non null label pointers.");
  }
  auto *fftwf_input_data = bolt_vector::fftwSegmentRowMajorActivations(
      inputs[0]->tensor()->getVector(index_in_batch), _rows, _columns);
  float* fftwf_output_data =
      static_cast<float*>(fftwf_malloc(_columns * _rows * sizeof(float)));

  #pragma omp critical
  {
    fftwf_plan plan =
        fftwf_plan_r2r_2d(_columns, _rows, fftwf_input_data, fftwf_output_data,
                        FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE);

    if (!plan) {
      fftwf_free(fftwf_input_data);
      fftwf_free(fftwf_output_data);
      throw std::runtime_error("Failed to create FFTW plan");
    }
    fftwf_execute(plan);

    fftwf_destroy_plan(plan);
  }

  std::memcpy(output->getVector(index_in_batch).activations, fftwf_output_data,
              _rows * _columns * sizeof(float));

  fftwf_free(fftwf_input_data);
  fftwf_free(fftwf_output_data);
}

void FFTMixer::backpropagate(ComputationList& inputs,
                             TensorPtr& output,
                             uint32_t index_in_batch) {
  assert(inputs.size() == 1 || inputs.size() == 2);

  auto *fftwf_input_data = bolt_vector::fftwSegmentRowMajorActivations(
      output->getVector(index_in_batch), _rows, _columns);
  float* fftwf_output_data =
      static_cast<float*>(fftwf_malloc(_columns * _rows * sizeof(float)));

  #pragma omp critical 
  {
    fftwf_plan plan =
        fftwf_plan_r2r_2d(_columns, _rows, fftwf_input_data, fftwf_output_data,
                          FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE);

    if (!plan) {
      fftwf_free(fftwf_input_data);
      fftwf_free(fftwf_output_data);
      throw std::runtime_error("Failed to create FFTW plan");
    }
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

  }

    std::memcpy(inputs[0]->tensor()->getVector(index_in_batch).gradients,
                fftwf_output_data, _rows * _columns * sizeof(float));

  fftwf_free(fftwf_input_data);
  fftwf_free(fftwf_output_data);
}

void FFTMixer::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

ComputationPtr FFTMixer::apply(ComputationPtr input) {
  if (input->dim() != _rows * _columns) {
    std::stringstream error;
    error << "Cannot apply FFT op with weight matrix of shape (" << _rows
          << ", " << _columns << ") to input tensor with dim " << input->dim()
          << ".";

    throw std::invalid_argument(error.str());
  }
  return Computation::make(shared_from_this(), {std::move(input)});
}

void FFTMixer::summary(std::ostream& summary,
                       const ComputationList& inputs,
                       const Computation* output) const {
  summary << "FFT(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();

  summary << "[rows=" << _rows << ", columns=" << _columns << "]";
}

template void FFTMixer::serialize(cereal::BinaryInputArchive&);
template void FFTMixer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void FFTMixer::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _rows, _columns);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::FFTMixer)
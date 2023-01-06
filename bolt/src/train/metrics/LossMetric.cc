#include "LossMetric.h"
#include <limits>

namespace thirdai::bolt::train::metrics {

LossMetric::LossMetric(nn::loss::LossPtr loss_fn)
    : _loss_fn(std::move(loss_fn)) {}

void LossMetric::record(uint32_t index_in_batch) {
  float sample_loss = _loss_fn->loss(index_in_batch);

  incrementAtomicFloat(_loss, sample_loss);
  _num_samples++;
}

void LossMetric::reset() {
  _loss = 0;
  _num_samples = 0;
}

float LossMetric::value() const { return _loss.load() / _num_samples.load(); }

float LossMetric::worst() const { return std::numeric_limits<float>::max(); }

bool LossMetric::betterThan(float a, float b) const { return a < b; }

std::string LossMetric::name() const { return "loss"; }

std::string LossMetric::outputName() const {
  // TODO(Nicholas): handle multiple outputs in loss, for example contrastive
  // loss.
  return _loss_fn->outputsUsed().at(0)->name();
}

}  // namespace thirdai::bolt::train::metrics
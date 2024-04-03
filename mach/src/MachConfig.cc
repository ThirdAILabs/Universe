#include "MachConfig.h"
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <mach/src/MachRetriever.h>

namespace thirdai::automl::mach {

std::shared_ptr<MachRetriever> MachConfig::build() const {
  return std::make_shared<MachRetriever>(*this);
}

data::StatePtr MachConfig::state() const {
  return data::State::make(
      dataset::mach::MachIndex::make(_n_buckets, _n_hashes),
      data::MachMemory::make(input_indices_column, input_values_column,
                             getIdCol(), bucket_column, _max_memory_ids,
                             _max_memory_samples_per_id));
}

float autotuneSparsity(size_t dim) {
  std::vector<std::pair<uint32_t, float>> sparsity_values = {
      {450, 1.0},    {900, 0.2},    {1800, 0.1},     {4000, 0.05},
      {10000, 0.02}, {20000, 0.01}, {1000000, 0.005}};

  for (const auto& [dim_threshold, sparsity] : sparsity_values) {
    if (dim < dim_threshold) {
      return sparsity;
    }
  }
  return sparsity_values.back().second;
}

bolt::ModelPtr MachConfig::model() const {
  auto input = bolt::Input::make(_text_feature_dim);

  auto hidden =
      bolt::Embedding::make(_emb_dim, _text_feature_dim, _emb_act, _emb_bias)
          ->apply(input);

  auto sparsity = autotuneSparsity(_n_buckets);

  auto output = bolt::FullyConnected::make(_n_buckets, hidden->dim(), sparsity,
                                           _output_act,
                                           /* sampling= */ nullptr,
                                           /* use_bias= */ _output_bias)
                    ->apply(hidden);

  auto labels = bolt::Input::make(_n_buckets);

  bolt::LossPtr loss;
  if (text::lower(_output_act) == "sigmoid") {
    loss = bolt::BinaryCrossEntropy::make(output, labels);
  } else if (text::lower(_output_act) == "softmax") {
    loss = bolt::CategoricalCrossEntropy::make(output, labels);
  } else {
    throw std::invalid_argument("Invalid output_act_func \"" + _output_act +
                                R"(". Choose one of "softmax" or "sigmoid".)");
  }

  return bolt::Model::make(
      {input}, {output}, {loss},
      // We need the hash based labels for training, but the actual
      // document/class ids to compute metrics. Hence we add two labels to the
      // model.
      {bolt::Input::make(std::numeric_limits<uint32_t>::max())});
}

data::TextTokenizerPtr MachConfig::textTransformation() const {
  return std::make_shared<data::TextTokenizer>(
      /* input_column= */ getTextCol(),
      /* output_indices= */ input_indices_column,
      /* output_values= */ input_values_column,
      /* tokenizer= */ getTextTokenizerFromString(_tokenizer),
      /* encoder= */ getTextEncoderFromString(_contextual_encoding),
      /* lowercase= */ _lowercase,
      /* dim= */ _text_feature_dim);
}

data::MachLabelPtr MachConfig::idToBucketsTransform() const {
  return std::make_shared<data::MachLabel>(getIdCol(), bucket_column);
}

}  // namespace thirdai::automl::mach